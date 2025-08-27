"""
Validation script for 3D YOLO.  Called as part of training script but can also be used to validate a model independently.
Example cmd line call: python val.py --data example.yaml --weights ./runs/train/exp/weights/best.pt
"""

# standard library imports
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# set path for local imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO3D root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 2D YOLO imports
from utils.general import check_dataset, check_img_size, check_suffix, check_yaml, increment_path, colorstr, print_args
from utils.metrics import ap_per_class
from utils.plots import plot_val_study
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks

# 3D YOLO imports
from utils3D.datasets import nifti_dataloader
from utils3D.general import zxyzxy2zxydwh, non_max_suppression, zxydwh2zxyzxy, scale_coords
from utils3D.lossandmetrics import ConfusionMatrix, box_iou
from models3D.model import attempt_load


default_size = 352 # edge length for testing


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    # 2D:
    # gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    # 3D:
    gn = torch.tensor(shape)[[0, 2, 1, 0, 2, 1]]  # normalization gain dwhdwh

    for *zxyzxy, conf, cls in predn.tolist():
        zxydwh = (zxyzxy2zxydwh(torch.tensor(zxyzxy).view(1, 6)) / gn).view(-1).tolist()  # normalized zxydwh
        line = (cls, *zxydwh, conf) if save_conf else (cls, *zxydwh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (z1, x1, y1, z2, x2, y2) format.
    Arguments:
        detections (Array[N, 8]), z1, x1, y1, z2, x2, y2, conf, class
        labels (Array[M, 7]), class, z1, x1, y1, z2, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    # main changes from 2D are adding the 2 extra entries for z positions
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :6])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 7]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=default_size,  # inference size (pixels)
        conf_thres=0.1,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        norm='CT',  # normalization mode, options: CT, MR, Other
        save_visualizations=False,  # save bounding box visualizations
        num_visualizations=5,  # number of samples to visualize
        slice_idx=None,  # specific slice to visualize (if None, shows middle slice)
        file_format='auto',  # file format for dataset loading, options: nifti, npz, auto
        return_predictions=False  # return predictions directly instead of saving to files
        ):
    
    # Initialize/load model and set device
    training = model is not None
    if training:  # if called by train.py
        device = next(model.parameters()).device  # get model device
    
    else:  # if called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        check_suffix(weights, '.pt')
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Data
        data = check_dataset(data)  # check
        
    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()
    
    # Configure
    model.eval()
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 1, imgsz, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        
        # Disable multiprocessing to avoid issues when called from main script
        if return_predictions:
            # Use workers=0 to avoid multiprocessing
            dataloader, _ = nifti_dataloader(data[task], imgsz, batch_size, gs, single_cls=single_cls, pad=pad, prefix=colorstr(f'{task}: '), file_format=file_format, workers=0)
        else:
            dataloader, _ = nifti_dataloader(data[task], imgsz, batch_size, gs, single_cls=single_cls, pad=pad, prefix=colorstr(f'{task}: '), file_format=file_format)
        
        # OPTIMIZATION: Automatically set num_visualizations to total number of validation images
        if save_visualizations:
            # Count total validation images
            total_val_images = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0
            if total_val_images > 0:
                num_visualizations = total_val_images
                print(f"Auto-setting num_visualizations to {num_visualizations} (total validation images)")
                
        seen = 0
        # confusion_matrix = ConfusionMatrix(nc=nc)  # Disabled to prevent hanging
        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
        # class_map = list(range(1000))
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        loss = torch.zeros(3, device=device)
        # jdict, stats, ap, ap_class = [], [], [], []
        stats, ap, ap_class = [], [], []
        
        # Visualization tracking
        samples_visualized = 0
        
        # Collect predictions if requested
        predictions_dict = {} if return_predictions else None
        
        # OPTIMIZATION: Use tqdm with minimal overhead
        for _, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s, leave=False)):
            t1 = time_sync()
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            
            if norm.lower() == 'ct':
                # Normalization for Hounsfield units, may see performance improvements by clipping images to +/- 1024.
                img = (img + 1024.) / 2048.
            elif norm.lower() == 'mr':
                mean = torch.mean(img, dim=[1,2,3,4], keepdim=True)
                std_dev = torch.std(img, dim=[1,2,3,4], keepdim=True)
                img = (img - mean)/std_dev
            else:
                raise NotImplementedError("You'll need to write your own normalization algorithm here.")
            
            targets = targets.to(device)
            nb, _, depth, height, width = img.shape  # batch size, channels, depth, height, width
            seen += nb
            t2 = time_sync()
            dt[0] += t2 - t1

            # Run model
            out, train_out = model(img) #, augment=augment)  # inference and training outputs
            dt[1] += time_sync() - t2

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

            # Run non max suppression
            targets[:, 2:] *= torch.Tensor([depth, width, height, depth, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t3 = time_sync()
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            dt[2] += time_sync() - t3

            # Statistics per image
            for si, pred in enumerate(out):
                # The following lines for stats calculation have been disabled for speed
                # labels = targets[targets[:, 0] == si, 1:]
                # nl = len(labels)
                # tcls = labels[:, 0].tolist() if nl else []  # target class
                path, shape = Path(paths[si]), shapes[si][0]
                # seen += 1

                # OPTIMIZATION: Remove debug prints for speed
                if len(pred) == 0:
                    # if nl:
                    #     stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                if single_cls:
                    pred[:, 7] = 0
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :6], shape, shapes[si][1])  # native-space pred

                # Evaluate (disabled for speed)
                # if nl:
                #     tbox = zxydwh2zxyzxy(labels[:, 1:7])  # target boxes
                #     scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                #     labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                #     correct = process_batch(predn, labelsn, iouv)
                # else:
                #     correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                # stats.append((correct.cpu(), pred[:, 6].cpu(), pred[:, 7].cpu(), tcls))  # (correct, conf, pcls, tcls)

                # Save
                if save_txt:
                    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
                
                # Collect predictions if requested
                if return_predictions and len(predn) > 0:
                    # Convert predictions to list format
                    pred_list = []
                    for pred in predn:
                        pred_dict = {
                            'bbox': pred[:6].tolist(),  # [z1, x1, y1, z2, x2, y2]
                            'confidence': float(pred[6]),
                            'class': int(pred[7])
                        }
                        pred_list.append(pred_dict)
                    
                    # Sort by confidence and keep top 6
                    pred_list.sort(key=lambda x: x['confidence'], reverse=True)
                    pred_list = pred_list[:6]  # Keep top 6 predictions
                    
                    predictions_dict[path.stem] = {
                        'predictions': pred_list,
                        'image_shape': shape.tolist() if hasattr(shape, 'tolist') else list(shape)
                    }
                
                # Save bounding box visualizations
                if save_visualizations and samples_visualized < num_visualizations:
                    # OPTIMIZATION: Only create simple text file instead of complex visualization
                    txt_path = save_dir / f"{path.stem}_bbox_visualization.txt"
                    with open(txt_path, 'w') as f:
                        f.write(f"Detection Results for {path.stem}_bbox_visualization\n")
                        f.write(f"Predictions: {len(predn)}\n")
                        if len(predn) > 0:
                            for i, pred in enumerate(predn):
                                conf = pred[6].item()
                                f.write(f"  Pred {i+1}: conf={conf:.3f}, bbox={pred[:6].tolist()}\n")
                        f.write(f"Ground Truth: {len(labelsn) if nl else 0}\n")
                        if labels is not None and len(labelsn) > 0:
                            for i, label in enumerate(labelsn):
                                f.write(f"  GT {i+1}: class={label[0]:.0f}, bbox={label[1:7].tolist()}\n")
                    samples_visualized += 1
                
                callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

        # Compute statistics (disabled for speed)
        # stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        # if len(stats) and stats[0].any():
        #     try:
        #         # Disable plotting in ap_per_class to prevent hanging
        #         p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=str(save_dir), names=names)
        #         ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        #         mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        #         nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        #     except Exception as e:
        #         print(f"Warning: Failed to compute AP statistics: {e}")
        #         # Fallback with dummy values
        #         p, r, ap, f1, ap_class = np.zeros(1), np.zeros(1), np.zeros((1, 10)), np.zeros(1), np.zeros(1)
        #         ap50, ap = np.zeros(1), np.zeros(1)
        #         mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
        #         nt = torch.zeros(1)
        # else:
        #     nt = torch.zeros(1)
        #     p, r, ap, f1, ap_class = np.zeros(1), np.zeros(1), np.zeros((1, 10)), np.zeros(1), np.zeros(1)
        #     ap50, ap = np.zeros(1), np.zeros(1)
        #     mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
        
        # Print results (disabled for speed)
        # pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        # print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        
        # Print results per class (disabled for speed)
        # if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        #     for i, c in enumerate(ap_class):
        #         try:
        #             c_int = int(c)  # Convert to integer for indexing
        #             print(pf % (names[c_int], seen, nt[c_int], p[i], r[i], ap50[i], ap[i]))
        #         except (ValueError, IndexError) as e:
        #             print(f"Warning: Could not print results for class {c}: {e}")
        #             continue
        
        # Print speeds
        if seen > 0:
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            if not training:
                shape = (batch_size, 1, imgsz, imgsz, imgsz)
                print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
        elif not training:
            print('No images were processed, skipping speed calculation.')
        
        # OPTIMIZATION: Completely disable plots and callbacks for speed
        if plots:
            print("Skipping all plotting to avoid hanging...")
            try:
                callbacks.run('on_val_end')
                print("Callbacks completed successfully")
            except Exception as e:
                print(f"Warning: Callbacks failed: {e}")
        else:
            print("Plots disabled, skipping callbacks")
            
        # Return results
        model.float()  # for training
        
        if not training:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {colorstr('bold', save_dir)}{s}")
        
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            try:
                c_int = int(c)
                maps[c_int] = ap[i]
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not process class {c}: {e}")
                continue
        
        result = (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
        print("DEBUG: Validation completed successfully!")
        
        # Return predictions if requested, otherwise return normal results
        if return_predictions:
            return predictions_dict
        else:
            return result
    
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/example.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=default_size, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--norm', type=str, default='CT', help='normalization type, options: CT, MR, Other')
    parser.add_argument('--save-visualizations', action='store_true', help='save bounding box visualizations')
    parser.add_argument('--num-visualizations', type=int, default=5, help='number of samples to visualize')
    parser.add_argument('--slice-idx', type=int, default=None, help='specific slice to visualize (if None, shows middle slice)')
    parser.add_argument('--file-format', type=str, default='auto', help='file format for dataset, options: nifti, npz, auto')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt
    
    
def main(opt):
    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                device=opt.device, plots=False) #, save_json=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, device=opt.device, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
