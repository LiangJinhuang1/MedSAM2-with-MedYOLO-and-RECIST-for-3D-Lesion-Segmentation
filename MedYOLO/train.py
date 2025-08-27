"""
Training script for 3D YOLO.
Example cmd line call: python train.py --data example.yaml --adam
"""

# standard library imports
import argparse
from copy import deepcopy
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm
import tempfile
import shutil

# set path for local imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO3D root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 2D YOLO imports
from utils.general import colorstr, labels_to_class_weights, increment_path, print_args, \
    check_yaml, check_file, get_latest_run, one_cycle, print_mutation, strip_optimizer, check_suffix
from utils.callbacks import Callbacks
from utils.torch_utils import select_device, de_parallel, EarlyStopping, ModelEMA, torch_distributed_zero_first, intersect_dicts
from utils.metrics import fitness
from utils.plots import plot_evolve

# 3D YOLO imports
from models3D.model import Model, attempt_load
from utils3D.datasets import nifti_dataloader, normalize_CT, normalize_MR
from utils3D.lossandmetrics import ComputeLossVF
from utils3D.anchors import nifti_check_anchors
from utils3D.general import check_dataset

import val
import matplotlib.pyplot as plt


def get_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1E9
        reserved = torch.cuda.memory_reserved() / 1E9
        max_allocated = torch.cuda.max_memory_allocated() / 1E9
        return allocated, reserved, max_allocated
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't provide detailed memory info, return placeholder values
        return 0, 0, 0
    else:
        # CPU mode
        return 0, 0, 0


def print_memory_stats(epoch, batch, allocated, reserved, max_allocated):
    """Print detailed memory statistics"""
    print(f"Epoch {epoch}, Batch {batch} - Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")


# Configuration
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# testing parameters, remove after dev
default_size = 350 # edge length for testing, below 350 the model can't process the data
default_epochs = 50
default_batch = 8  # Reduced from 8 to 4 for memory optimization


def check_memory_warning(allocated, reserved, max_allocated, batch_size):
    """Check if memory usage is high and suggest optimizations"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1E9
        memory_usage_ratio = reserved / total_memory
        
        if memory_usage_ratio > 0.85:
            print(f"⚠️  WARNING: High memory usage detected!")
            print(f"   Memory usage: {memory_usage_ratio:.1%} ({reserved:.2f}GB / {total_memory:.2f}GB)")
            print(f"   Current batch size: {batch_size}")
            if batch_size > 2:
                suggested_batch = max(1, batch_size // 2)
                print(f"   Suggestion: Reduce batch size to {suggested_batch}")
            else:
                print(f"   Suggestion: Consider reducing image size or model complexity")
            print(f"   Max memory used: {max_allocated:.2f}GB")
            return True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS memory management is handled by the system
        return False
    else:
        # CPU mode - no GPU memory concerns
        return False


def save_checkpoint_safely(ckpt, filepath, max_retries=3):
    """
    Safely save checkpoint with atomic write and error handling.
    
    Args:
        ckpt: Checkpoint dictionary to save
        filepath: Path where to save the checkpoint
        max_retries: Maximum number of retry attempts
    """
    filepath = Path(filepath)
    
    # Ensure the parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            # Create temporary file in the same directory
            temp_file = filepath.parent / f"temp_checkpoint_{attempt}.pt"
            
            # Save to temporary file first
            torch.save(ckpt, temp_file)
            
            # Atomic move to final location
            shutil.move(str(temp_file), str(filepath))
            
            print(f"Checkpoint saved successfully to {filepath}")
            return True
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to save checkpoint: {e}")
            
            # Clean up temporary file if it exists
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
            
            if attempt == max_retries - 1:
                print(f"Failed to save checkpoint after {max_retries} attempts")
                return False
            
            # Wait a bit before retrying
            time.sleep(1)
    
    return False


def create_lightweight_checkpoint(model, ema, optimizer, epoch, best_fitness):
    """
    Create a lightweight checkpoint to reduce memory usage and file size.
    
    Args:
        model: The model to save
        ema: EMA model
        optimizer: Optimizer state
        epoch: Current epoch
        best_fitness: Best fitness value
    
    Returns:
        Dictionary containing the lightweight checkpoint
    """
    # Create a lightweight checkpoint
    ckpt = {
        'epoch': epoch,
        'best_fitness': best_fitness,
        'model': deepcopy(de_parallel(model)).half(),  # Save in half precision
        'ema': deepcopy(ema.ema).half(),  # Save EMA in half precision
        'updates': ema.updates,
        'optimizer': optimizer.state_dict()
    }
    
    # Clear gradients to save memory
    for param in ckpt['model'].parameters():
        if param.grad is not None:
            param.grad.data.zero_()
    
    return ckpt


def train(hyp, opt, device, callbacks):
    # Device setup and memory optimization
    print(f"Using device: {device}")
    
    # Memory optimization setup for GPU
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        # Set memory optimization flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Enable gradient checkpointing for memory efficiency
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Additional CUDA optimizations for speed
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    elif device.type == 'mps':
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    else:
        print("Using CPU for training")
    
    # parsing the arguments
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, norm = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.norm
    
    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load model hyps dict
    
    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None
   
    # Config
    plots = False
    cuda = device.type != 'cpu'
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    
    # Model
    check_suffix(weights, '.pt')
    pretrained = weights.endswith('.pt')
    if pretrained:
        try:
            # Try with weights_only=True first (PyTorch 2.6+ default)
            ckpt = torch.load(weights, map_location=device, weights_only=True)  # load checkpoint
        except Exception:
            # Fall back to weights_only=False for compatibility with older checkpoints
            ckpt = torch.load(weights, map_location=device, weights_only=False)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=1, nc=nc, anchors=hyp.get('anchors')).to(device) # create model
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
    else:        
        model = Model(cfg = cfg, ch=1, nc=nc, anchors=hyp.get('anchors')).to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for memory optimization")
    
    # loads from models folder
    with open(data, errors='ignore') as f:
        data_dict = yaml.safe_load(f)  # model dict

    nc = int(data_dict['nc'])  # number of classes

    train_path, val_path = data_dict['train'], data_dict['val']
    
    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    
    # Image size
    imgsz = opt.imgsz
    stride = model.stride
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm3d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)   
    
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    del g0, g1, g2
    
    # Scheduler
    lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None
    
    # Resume
    start_epoch, best_fitness = 0, 0.0
    if resume:
        # Load checkpoint for resuming training
        if pretrained:
            start_epoch = ckpt['epoch'] + 1
            best_fitness = ckpt['best_fitness']
            print(f"Resuming training from epoch {start_epoch} (best fitness: {best_fitness:.4f})")
            
            # Load optimizer state
            if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    print("Optimizer state restored")
                except Exception as e:
                    print(f"Warning: Failed to load optimizer state: {e}")
                    print("Continuing with fresh optimizer state")
            else:
                print("No optimizer state found in checkpoint, using fresh optimizer")
            
            # Load EMA state
            if ema and 'ema' in ckpt and ckpt['ema'] is not None:
                try:
                    if isinstance(ckpt['ema'], dict):
                        ema.ema.load_state_dict(ckpt['ema'])
                    else:
                        # If ema is a model object, load its state dict
                        ema.ema.load_state_dict(ckpt['ema'].state_dict())
                    ema.updates = ckpt.get('updates', 0)
                    print("EMA state restored")
                except Exception as e:
                    print(f"Warning: Failed to load EMA state: {e}")
                    print("Continuing with fresh EMA state")
            else:
                print("No EMA state found in checkpoint, using fresh EMA")
            
            # Load scheduler state
            if 'scheduler' in ckpt and ckpt['scheduler'] is not None:
                try:
                    scheduler.load_state_dict(ckpt['scheduler'])
                    print("Scheduler state restored")
                except Exception as e:
                    print(f"Warning: Failed to load scheduler state: {e}")
                    print("Continuing with fresh scheduler state")
            else:
                print("No scheduler state found in checkpoint, using fresh scheduler")
        else:
            print("Warning: No checkpoint found for resume, starting from epoch 0")
    
    # Trainloader
    train_loader, train_dataset = nifti_dataloader(train_path,
                                                   imgsz=imgsz,
                                                   batch_size=batch_size,
                                                   hyp=hyp,
                                                   single_cls=single_cls,
                                                   stride=stride,
                                                   rank=LOCAL_RANK,
                                                   workers=workers,
                                                   augment=True,
                                                   file_format=opt.file_format,
                                                   cache_dir=opt.cache_dir)
    mlc = int(np.concatenate(train_dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
    
    # Create val_loader on all ranks to ensure synchronized state before DDP initialization
    val_loader = nifti_dataloader(val_path,
                                  imgsz=imgsz,
                                  batch_size=batch_size,
                                  stride=stride,
                                  single_cls=single_cls,
                                  workers=workers,
                                  file_format=opt.file_format,
                                  cache_dir=opt.cache_dir,
                                  rank=LOCAL_RANK)[0]

    # Process 0 - Rank-specific setup
    if RANK in [-1, 0]:
        if not resume and not opt.noautoanchor:
            # Anchors are checked only on rank 0
            nifti_check_anchors(train_dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
        
        callbacks.run('on_pretrain_routine_end')
        
    # Ensure model is consistent across all ranks before DDP initialization
    if not resume:
        model.half().float()

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        
    # Model parameters
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps), default is 3
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / default_size) ** 3 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    
    # Start training
    # t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    # maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLossVF(model)  # init Varifocal loss class
    
    # Training loss tracking for plotting
    training_losses = {
        'epochs': [],
        'box_losses': [],
        'obj_losses': [],
        'cls_losses': [],
        'total_losses': []
    }

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        print(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        
        # Clear cache at start of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # train loop
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            # Normalization
            if norm.lower() == 'ct':
                imgs = normalize_CT(imgs.to(device, non_blocking=True).float())  # int to float32, -1024-1024 to 0.0-1.0
            elif norm.lower() == 'mr':
                imgs = normalize_MR(imgs.to(device, non_blocking=True).float())  # int to float32, mean 0 std dev
            else:
                raise NotImplementedError("You'll need to write your own normalization algorithm here.")

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
                        
            # Forward
            try:
                with amp.autocast(enabled=cuda):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    del pred

                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM error detected at epoch {epoch}, batch {i}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("Memory cleared. Consider reducing batch size further.")
                    raise
                else:
                    raise

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
        
            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                allocated, reserved, max_allocated = get_memory_usage()
                mem = f'{reserved:.3g}G'  # (GB)
                
                # Check for memory warnings every 10 batches
                if i % 10 == 0:
                    check_memory_warning(allocated, reserved, max_allocated, batch_size)
                
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, False)
            del imgs, targets
            
            # Optimized memory cleanup - less frequent for better performance
            if torch.cuda.is_available() and i % 20 == 0:  # Clear cache every 20 batches instead of 5
                torch.cuda.empty_cache()
            # end batch ------------------------------------------------------------------------------------------------
            
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        
        # Record training losses for plotting
        if RANK in [-1, 0]:
            training_losses['epochs'].append(epoch)
            training_losses['box_losses'].append(mloss[0].item())
            training_losses['obj_losses'].append(mloss[1].item())
            training_losses['cls_losses'].append(mloss[2].item())
            training_losses['total_losses'].append(mloss.sum().item())
        
        # Validation and Checkpointing (on main process)
        if RANK in [-1, 0]:
            # Validation
            fi = 0.0 # Default fitness
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval:
                # Use lower confidence threshold for early training
                conf_thres = 0.01 if epoch < 5 else 0.1  # Start with very low threshold
                print(f"Validation with confidence threshold: {conf_thres}")
                
                results, _, _ = val.run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE * 2,
                                           imgsz=imgsz,
                                           model=ema.ema,
                                           single_cls=single_cls,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss,
                                           norm=norm,
                                           conf_thres=conf_thres,
                                           verbose=True)  # Enable verbose for debugging
                
                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                if fi > best_fitness:
                    best_fitness = fi
            
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save Checkpoint
            if not nosave and not evolve:
                ckpt = create_lightweight_checkpoint(model, ema, optimizer, epoch, best_fitness)
                
                # Save last checkpoint
                save_checkpoint_safely(ckpt, last, opt.checkpoint_retries)
                
                # Save best checkpoint
                if best_fitness == fi and not noval:
                    save_checkpoint_safely(ckpt, best, opt.checkpoint_retries)
                
                # Save periodic checkpoint if enabled - REMOVED TO SAVE SPACE
                # if opt.save_period > 0 and (epoch + 1) % opt.save_period == 0:
                #     save_checkpoint_safely(ckpt, w / f'epoch{epoch+1}.pt', opt.checkpoint_retries)
                
                del ckpt
        
        # Early stopping
        if RANK == -1 and stopper(epoch=epoch, fitness=fi):
            break
        # end epoch ----------------------------------------------------------------------------------------------------
    
    # Final cleanup and validation
    if RANK in [-1, 0]:
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best and not noval:
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=attempt_load(f, device).half(),
                                            iou_thres=0.60,
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            verbose=True,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss,
                                            norm=norm)  # val best model with plots

        callbacks.run('on_train_end', last, best, plots, epoch, results)

    torch.cuda.empty_cache()
    
    # Plot training losses if we have data
    if RANK in [-1, 0] and training_losses['epochs']:
        plot_training_losses(training_losses, save_dir)
    
    return results


def plot_training_losses(training_losses, save_dir):
    """Plot training loss trends and save to file."""
    try:
        epochs = training_losses['epochs']
        box_losses = training_losses['box_losses']
        obj_losses = training_losses['obj_losses']
        cls_losses = training_losses['cls_losses']
        total_losses = training_losses['total_losses']
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('MedYOLO Training Loss Trends', fontsize=16)
        
        # Plot individual losses
        ax1.plot(epochs, box_losses, 'b-', label='Box Loss', linewidth=2)
        ax1.set_title('Box Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(epochs, obj_losses, 'r-', label='Object Loss', linewidth=2)
        ax2.set_title('Object Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.plot(epochs, cls_losses, 'g-', label='Class Loss', linewidth=2)
        ax3.set_title('Class Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot total loss
        ax4.plot(epochs, total_losses, 'purple', label='Total Loss', linewidth=2)
        ax4.set_title('Total Loss (Box + Object + Class)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = Path(save_dir) / 'training_loss_components.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training loss plot saved to: {plot_path}")
        
        # Create combined plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, box_losses, 'b-', label='Box Loss', linewidth=2)
        plt.plot(epochs, obj_losses, 'r-', label='Object Loss', linewidth=2)
        plt.plot(epochs, cls_losses, 'g-', label='Class Loss', linewidth=2)
        plt.plot(epochs, total_losses, 'purple', label='Total Loss', linewidth=3)
        
        plt.title('MedYOLO Training Loss Trends', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Save combined plot
        combined_plot_path = Path(save_dir) / 'training_loss_combined.png'
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        print(f"Combined training loss plot saved to: {combined_plot_path}")
        
        # Print final loss values
        print(f"\nFinal Training Losses:")
        print(f"  Box Loss: {box_losses[-1]:.4f}")
        print(f"  Object Loss: {obj_losses[-1]:.4f}")
        print(f"  Class Loss: {cls_losses[-1]:.4f}")
        print(f"  Total Loss: {total_losses[-1]:.4f}")
        
        plt.close('all')  # Close all plots to free memory
        
    except Exception as e:
        print(f"Warning: Failed to create training loss plots: {e}")
        print("Training completed successfully, but loss plots could not be generated.")


def parse_opt(known=False):
    # parses the options in the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models3D/yolo3Ds.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/example.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyps/hyp.finetune.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=default_epochs)
    parser.add_argument('--batch-size', type=int, default=default_batch, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=default_size, help='train, val image size (pixels)')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='device to use (cuda device, i.e. 0 or 0,1,2,3, or cpu, or mps, or empty for auto-detect)')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--workers', type=int, default=16, help='maximum number of dataloader workers')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=200, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--norm', type=str, default='CT', help='normalization type, options: CT, MR, Other')
    parser.add_argument('--file-format', type=str, default='auto', help='file format for dataset, options: nifti, npz')
    parser.add_argument('--cache-dir', type=str, default=None, help='path to directory for caching dataset files')
    parser.add_argument('--no-checkpoint', action='store_true', help='disable checkpoint saving completely')
    parser.add_argument('--checkpoint-retries', type=int, default=3, help='number of retries for checkpoint saving')
    parser.add_argument('--save-only-last', action='store_true', help='save only the last checkpoint and remove old ones')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)
        
    # Resume
    if opt.resume and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    
    # Clean up corrupted checkpoints before starting
    if RANK in [-1, 0]:
        print("Checking for corrupted checkpoint files...")
        cleanup_corrupted_checkpoints(opt.save_dir)
        
        # Clean up old checkpoints to save space (keep only last.pt)
        if opt.save_only_last:
            print("Cleaning up old checkpoints to save space...")
            cleanup_old_checkpoints(opt.save_dir, keep_files=['last.pt'])
        
    # Device selection and DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    
    # Print device information
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1E9:.1f} GB")
    elif device.type == 'mps':
        print("MPS (Metal Performance Shaders) available")
    else:
        print("Using CPU for training")
    
    if LOCAL_RANK != -1:
        # DDP mode - only for CUDA devices
        if device.type == 'cuda':
            assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
            assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
            torch.cuda.set_device(LOCAL_RANK)
            device = torch.device('cuda', LOCAL_RANK)
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        else:
            print("Warning: DDP mode is not supported for non-CUDA devices. Running in single-device mode.")
            # Don't modify global variables, just skip DDP setup
            pass
        
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
    
    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            dist.destroy_process_group()
    
    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'max_zoom': (0, 1.0, 2.0), # maximum zoom factor
                'min_zoom': (0, 0.4, 1.0), # minimum zoom factor
                'prob_zoom': (0, 0.3, 0.7), # probability of zoom augmentation
                'prob_cutout': (0, 0.3, 0.7), # probability of cutout augmentation
        }
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists
    
        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate
   
            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits
                
            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)

            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)
            
        # Plot results
        plot_evolve(evolve_csv)
        print(f'Hyperparameter evolution finished\n'
              f"Results saved to {colorstr('bold', save_dir)}\n"
              f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')
       
        
def run(**kwargs):
    # Usage: import train; train.run(data='example.yaml', imgsz=350, weights='yolo3Ds.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


def cleanup_corrupted_checkpoints(save_dir):
    """
    Clean up corrupted checkpoint files that might cause serialization errors.
    
    Args:
        save_dir: Directory containing checkpoint files
    """
    save_dir = Path(save_dir)
    
    # Check for corrupted checkpoint files
    checkpoint_files = list(save_dir.glob("*.pt"))
    
    for ckpt_file in checkpoint_files:
        try:
            # Try to load the checkpoint to check if it's corrupted
            try:
                torch.load(ckpt_file, map_location='cpu', weights_only=True)
            except Exception:
                torch.load(ckpt_file, map_location='cpu', weights_only=False)
            print(f"Checkpoint {ckpt_file} is valid")
        except Exception as e:
            print(f"Corrupted checkpoint detected: {ckpt_file}")
            print(f"Error: {e}")
            
            # Create backup of corrupted file
            backup_file = ckpt_file.with_suffix('.pt.corrupted')
            try:
                shutil.move(str(ckpt_file), str(backup_file))
                print(f"Corrupted checkpoint moved to {backup_file}")
            except Exception as move_error:
                print(f"Failed to move corrupted checkpoint: {move_error}")
                # Try to delete the corrupted file
                try:
                    ckpt_file.unlink()
                    print(f"Deleted corrupted checkpoint: {ckpt_file}")
                except Exception as del_error:
                    print(f"Failed to delete corrupted checkpoint: {del_error}")


def cleanup_old_checkpoints(save_dir, keep_files=['best.pt', 'last.pt']):
    """
    Clean up old checkpoint files, keeping only specified files.
    
    Args:
        save_dir: Directory containing checkpoint files
        keep_files: List of checkpoint files to keep
    """
    save_dir = Path(save_dir)
    weights_dir = save_dir / 'weights'
    
    if weights_dir.exists():
        # Clean up checkpoint files in weights directory
        for old_ckpt in weights_dir.glob("*.pt"):
            if old_ckpt.name not in keep_files:
                try:
                    old_ckpt.unlink()
                    print(f"Cleaned up old checkpoint: {old_ckpt.name}")
                except Exception as e:
                    print(f"Failed to remove old checkpoint {old_ckpt.name}: {e}")
    
    # Also clean up any checkpoint files in the main save directory
    for old_ckpt in save_dir.glob("*.pt"):
        if old_ckpt.name not in keep_files:
            try:
                old_ckpt.unlink()
                print(f"Cleaned up old checkpoint: {old_ckpt.name}")
            except Exception as e:
                print(f"Failed to remove old checkpoint {old_ckpt.name}: {e}")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
   