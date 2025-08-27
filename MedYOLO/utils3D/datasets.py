"""
Dataloaders and dataset utils for nifti datasets for YOLO3D
"""

# standard library imports
import nibabel as nib
import numpy as np
from pathlib import Path
import glob
import os
from multiprocessing.pool import Pool
from tqdm import tqdm
from itertools import repeat
from typing import List
import torch
from torch.utils.data import Dataset
import torch.distributed as dist

# 2D YOLO imports
from utils.torch_utils import torch_distributed_zero_first
from utils.datasets import InfiniteDataLoader, get_hash

# 3D YOLO imports
from utils3D.general import zxyzxy2zxydwhn, zxydwhn2zxyzxy
from utils3D.augmentations import tensor_cutout, random_zoom


# Configuration
IMG_FORMATS = ['nii', 'nii.gz', 'npz']  # acceptable image suffixes, note nii.gz compatible by checking for presence of 'nii' in -2 place
NUM_THREADS = min(4, os.cpu_count())  # Reduced from 8 to 4 for memory optimization
default_size = 352 # edge length for testing


def file_lister_train(parent_dir: List[str], prefix=''):
    """Takes a parent directory or list of parent directories and
    looks for files within those directories.  Output organized to fit
    YOLO training requirements.

    Args:
        parent_dir (List[str] or str): Folders to be searched.  Text files allowed.
        prefix (str, optional): Prefix for error messages. Defaults to ''.

    Raises:
        Exception: If parent_dir is neither a directory nor file.

    Returns:
        file_list (List[str]): a list of paths to the files found.
        p (pathlib.PosixPath): the path to the parent directory, for caching purposes
    """

    file_list = []
    for p in parent_dir if isinstance(parent_dir, list) else [parent_dir]:
        p = Path(p)
        if p.is_dir():  # dir
            file_list += glob.glob(str(p / '**' / '*.*'), recursive=True)
        elif p.is_file():  # file
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                file_list += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                # file_list += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
        else:
            raise Exception(f'{prefix}{p} does not exist')
    return file_list, p


def file_lister_detect(parent_dir: str):
    """Takes a parent directory and looks for files within those directories.
    Output organized to fit YOLO inference requirements.

    Args:
        parent_dir (str): parent folder to search for files.

    Raises:
        Exception: if parent_dir is not a file, directory, or glob search pattern.

    Returns:
        files (List[str]): a list of paths to the files found.
    """
    p = str(Path(parent_dir).resolve())
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return files


class LoadNiftis(Dataset):
    """YOLO3D Pytorch Dataset for inference. Supports NIfTI and NPZ files.
    Args:
        file_format (str): 'auto', 'nifti', or 'npz'. If 'auto', detects format from file extensions. Default: 'auto'.
    """
    def __init__(self, path: str, img_size=default_size, stride=32, file_format='auto'):
        """Initialization for the inference Dataset

        Args:
            path (str): parent directory for the Dataset's files
            img_size (int, optional): edge length for the cube input will be reshaped to. Defaults to default_size (currently 350).
            stride (int, optional): model stride, used for resizing and augmentation, currently unimplemented. Defaults to 32.
            file_format (str): 'auto', 'nifti', or 'npz'. If 'auto', detects format from file extensions. Default: 'auto'.
        """
        
        # Find files in the given path and filter to leave only .nii and .nii.gz files in the list
        files = file_lister_detect(path)
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS or x.split('.')[-2].lower() in IMG_FORMATS]

        self.nf = len(images)
        self.files = images
        self.img_size = img_size
        self.stride = stride

        # Auto-detect file format if not specified
        if file_format == 'auto':
            # Check if all files are npz
            npz_files = [f for f in self.files if f.endswith('.npz')]
            nifti_files = [f for f in self.files if f.endswith('.nii') or f.endswith('.nii.gz')]
            
            if len(npz_files) == len(self.files):
                self.file_format = 'npz'
            elif len(nifti_files) == len(self.files):
                self.file_format = 'nifti'
            else:
                # Mixed formats or other formats
                raise Exception(f'Mixed file formats detected. Please specify file_format explicitly or use consistent file types.')
        else:
            self.file_format = file_format

        assert self.nf > 0, f'No images found in {path}. Supported formats are: {IMG_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # Iterate through the list of files
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read current image
        self.count += 1
        # img0, affine = open_nifti(path)
        img0, _ = open_image(path, self.file_format)
        assert img0 is not None, 'Image Not Found ' + path
        print(f'\nimage {self.count}/{self.nf} {path}: ', end='')

        # Reshape image to fit model requirements
        img = transpose_nifti_shape(img0)
        img = change_nifti_size(img, self.img_size)

        return path, img, img0

    def __len__(self):
        return self.nf  # number of files


def open_npz(filepath: str):
    """Reads a .npz file and converts it to torch tensors.
    Assumes the npz contains 'imgs' and 'gts' arrays.
    Returns: (image_tensor, mask_tensor, None)
    """
    # Memory optimization: Load with mmap_mode for large files
    try:
        arr = np.load(filepath, mmap_mode='r')  # Memory-mapped loading
        img_data = arr['imgs']  # 3D lesion image, shape=(Z,Y,X)
        mask_data = arr['gts']  # 3D lesion mask, shape=(Z,Y,X)
        
        # Convert to torch tensors with memory-efficient dtype
        img_tensor = torch.tensor(img_data, dtype=torch.float32)  # Use float32 instead of float64
        mask_tensor = torch.tensor(mask_data, dtype=torch.float32)
        
        # Close the memory-mapped file
        arr.close()
        
        return img_tensor, mask_tensor, None  # No affine for npz
    except Exception as e:
        # Fallback to regular loading if mmap fails
        arr = np.load(filepath)
        img_data = arr['imgs']
        mask_data = arr['gts']
        
        img_tensor = torch.tensor(img_data, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_data, dtype=torch.float32)
        
        return img_tensor, mask_tensor, None


def open_image(filepath: str, file_format: str = 'nifti'):
    """Generic image loader for nifti or npz files."""
    if file_format == 'npz' or filepath.endswith('.npz'):
        return open_npz(filepath)
    else:
        img, affine = open_nifti(filepath)
        return img, None, affine  # Return None for mask when using nifti


def open_nifti(filepath: str):
    """Reads a nifti file and converts it to a torch tensor

    Args:
        filepath (str): Path to the nifti file

    Returns:
        nifti (torch.tensor): Tensor containing the nifti image data
        nifti_affine: affine array for the nifti
    """
    nifti = nib.load(filepath)
    nifti_affine = nifti.affine
    # nifti_array = np.array(nifti.dataobj)
    # assert nifti_array is not None, 'Image Not Found ' + filepath
    # nifti_tensor = torch.tensor(nifti_array, dtype=torch.float)
    # return nifti_tensor, nifti_affine
    nifti = np.array(nifti.dataobj)
    assert nifti is not None, 'Image Not Found ' + filepath
    nifti = torch.tensor(nifti, dtype=torch.float)
    return nifti, nifti_affine


def transpose_nifti_shape(nifti_tensor: torch.Tensor):
    """Reshapes the tensor from height, width, depth order to depth, height, width
    to make it compatible with torch convolutions.

    Args:
        nifti_tensor (torch.tensor): tensor to be reshaped

    Returns:
        nifti_tensor (torch.tensor): reshaped tensor
    """
    nifti_tensor = torch.transpose(nifti_tensor, 0, 2)
    nifti_tensor = torch.transpose(nifti_tensor, 1, 2)
    return nifti_tensor


def change_nifti_size(nifti_tensor: torch.Tensor, new_size: int):
    """Resizes a 3D tensor to a cube with edge length new_size.
    Also adds the channel dimension.

    Args:
        nifti_tensor (torch.Tensor): The tensor to be resized
        new_size (int): The edge length for the resized, cubic tensor

    Returns:
        nifti_tensor (torch.tensor): Resized, cubic tensor
    """
    # add channel dimension for compatibility with later code
    nifti_tensor = torch.unsqueeze(nifti_tensor, 0)
    # add batch dimension for functional interpolate
    nifti_tensor = torch.unsqueeze(nifti_tensor, 0)
    # resize image to a cube of size new_size
    nifti_tensor = torch.nn.functional.interpolate(nifti_tensor, size=(new_size, new_size, new_size), mode='trilinear', align_corners=False)
    # remove batch dimension for compatibility with later code
    nifti_tensor = torch.squeeze(nifti_tensor, 0)
    return nifti_tensor


def normalize_CT(imgs):
    """Normalizes 3D CTs in Hounsfield Units (+/- 1024) to within 0 and 1.

    Args:
        imgs (torch.tensor): unnormalized model input

    Returns:
        imgs (torch.tensor): normalized model input
    """
    imgs = (imgs + 1024.) / 2048.0  # int to float32, -1024-1024 to 0.0-1.0
    return imgs


def normalize_MR(imgs):
    """Volume normalizes 3D MR images to mean 0 and standard deviation 1.

    Args:
        imgs (torch.tensor): unnormalized model input

    Returns:
        imgs (torch.tensor): normalized model input
    """
    means = torch.mean(imgs, dim=[1,2,3,4], keepdim=True)
    std_devs = torch.std(imgs, dim=[1,2,3,4], keepdim=True)
    imgs = (imgs - means)/std_devs
    return imgs


def mask_to_yolo_labels(mask_tensor, img_shape):
    """Convert 3D mask tensor to YOLO format labels.
    Args:
        mask_tensor: 3D tensor with shape (Z, Y, X)
        img_shape: tuple of (D, H, W) for the image
    Returns:
        labels: numpy array with shape (N, 7) where N is number of objects
                Format: [class, z_center, x_center, y_center, depth, width, height] (normalized)
    """
    # Find connected components in the mask
    mask_np = mask_tensor.numpy()
    from scipy import ndimage
    labeled_mask, num_features = ndimage.label(mask_np)
    
    labels = []
    for i in range(1, num_features + 1):
        # Get coordinates of this object
        coords = np.where(labeled_mask == i)
        if len(coords[0]) == 0:
            continue
            
        # Calculate bounding box
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
        # Convert to YOLO format (center coordinates, normalized)
        D, H, W = img_shape
        z_center = (z_min + z_max) / 2.0 / D
        y_center = (y_min + y_max) / 2.0 / H
        x_center = (x_min + x_max) / 2.0 / W
        
        depth = (z_max - z_min + 1) / D
        height = (y_max - y_min + 1) / H
        width = (x_max - x_min + 1) / W
        
        # Ensure normalized coordinates are within [0, 1]
        z_center = np.clip(z_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        x_center = np.clip(x_center, 0, 1)
        depth = np.clip(depth, 0, 1)
        height = np.clip(height, 0, 1)
        width = np.clip(width, 0, 1)
        
        # Class 0 (assuming single class)
        labels.append([0, z_center, x_center, y_center, depth, width, height])
    
    return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 7), dtype=np.float32)


class LoadNiftisAndLabels(Dataset):
    """YOLO3D Pytorch Dataset for training. Supports NIfTI and NPZ files.
    Args:
        file_format (str): 'auto', 'nifti', or 'npz'. If 'auto', detects format from file extensions. Default: 'auto'.
    """
    cache_version = 0.63  # Unified cache version

    def __init__(self, path, img_size=default_size, batch_size=4, augment=False, hyp=None, single_cls=False,
                 stride=32, pad=0.0, prefix='', file_format='auto', cache_dir=None):
        self.img_size = img_size
        self.stride = stride
        self.path = path
        self.augment = augment
        self.hyp = hyp

        # --- Determine file format and cache path ---
        path = Path(path)
        if file_format == 'auto':
            # This is a quick check. It assumes if any .npz files are found, the whole dataset is npz.
            # A more robust check might be needed for mixed datasets, but this is fast.
            temp_files, _ = file_lister_train(str(path), prefix)
            self.file_format = 'npz' if any(f.endswith('.npz') for f in temp_files) else 'nifti'
        else:
            self.file_format = file_format
        
        if cache_dir:
            cache_path = Path(cache_dir) / f"{path.name}.{self.file_format}.cache"
        else:
            # Try to save cache next to data, fallback to current dir if no perms
            try:
                (path.parent / '.cache_test').touch()
                (path.parent / '.cache_test').unlink()
                cache_path = path.with_suffix(f'.{self.file_format}.cache')
            except OSError:
                cache_path = Path(f'./{path.name}.{self.file_format}.cache')

        print(f"{prefix} Using cache file: {cache_path}")

        # --- Synchronized Cache Creation/Loading ---
        RANK = int(os.getenv('RANK', -1))
        with torch_distributed_zero_first(RANK):
            if not cache_path.is_file() or self.check_cache_validity(cache_path) is False:
                print(f"{prefix}Cache not found or invalid. Creating new cache...")
                self.create_unified_cache(path, cache_path, prefix)

        # All ranks load the cache
        print(f"{prefix}Rank {RANK if RANK!=-1 else 0} loading cache from {cache_path}")
        try:
            cache = torch.load(cache_path, weights_only=False)
            assert cache['version'] == self.cache_version, "Cache version mismatch. Please delete cache file and re-run."
            self.labels = cache['labels']
            self.shapes = cache['shapes']
            self.segments = cache['segments']
            self.img_files = cache['img_files']
            print(f"{prefix}Rank {RANK if RANK!=-1 else 0} successfully loaded {len(self.img_files)} files from cache.")
        except Exception as e:
            raise Exception(f"Rank {RANK if RANK!=-1 else 0} FAILED to load cache '{cache_path}'. "
                            f"Delete the cache file and try again. Error: {e}")

        # --- Final Dataset Setup ---
        n = len(self.shapes)
        self.batch = np.floor(np.arange(n) / batch_size).astype(int)
        self.n = n
        self.indices = range(n)

        # Update labels if single_cls
        if single_cls:
            for i in range(len(self.labels)):
                self.labels[i][:, 0] = 0
        
        self.imgs, self.img_npy = [None] * n, [None] * n

    def check_cache_validity(self, cache_path):
        """Checks if the cache file is valid without loading the whole thing."""
        try:
            cache = torch.load(cache_path)
            return cache.get('version') == self.cache_version
        except Exception:
            return False

    def create_unified_cache(self, path, cache_path, prefix):
        """Scans for image files, processes them, and creates a single cache file with incremental saving for npz."""
        part_cache_path = cache_path.with_suffix('.part')

        print(f"{prefix}Scanning {path} for images...")
        all_files, _ = file_lister_train(str(path), prefix)
        
        if self.file_format == 'npz':
            img_files = sorted([x for x in all_files if x.endswith('.npz')])
        else: # nifti
            img_files = sorted([x for x in all_files if x.endswith(('.nii', '.nii.gz'))])

        assert img_files, f"{prefix}No images found for format '{self.file_format}' in {path}"
        print(f"{prefix}Found {len(img_files)} total images.")

        labels_list, shapes_list, segments_list = [], [], []
        processed_files = set()

        # Resume from partial cache if it exists and is valid (only for NPZ)
        if self.file_format == 'npz' and part_cache_path.is_file():
            try:
                print(f"{prefix}Resuming from partial cache: {part_cache_path}")
                part_cache = torch.load(part_cache_path)
                if part_cache.get('version') == self.cache_version:
                    labels_list = part_cache['labels']
                    shapes_list = part_cache['shapes']
                    segments_list = part_cache['segments']
                    processed_files = set(part_cache['img_files'])
                    print(f"{prefix}Resumed processing. {len(processed_files)} files already cached.")
                else:
                    print(f"{prefix}Partial cache version mismatch. Starting fresh.")
            except Exception as e:
                print(f"{prefix}Could not load partial cache ({e}). Starting fresh.")
        
        files_to_process = [f for f in img_files if f not in processed_files]
        if not files_to_process:
            print(f"{prefix}All files are already processed in cache.")
        else:
            print(f"{prefix}{len(files_to_process)} files remaining to be processed.")

        if self.file_format == 'npz':
            save_interval = 200
            pbar = tqdm(files_to_process, desc=f'{prefix}Processing npz files for cache')
            # For NPZ files, we generate labels directly from the mask.
            for i, img_file in enumerate(pbar):
                try:
                    img, mask, _ = open_image(img_file, 'npz')
                    # NPZ files from this project seem to be in Z,Y,X format already. No transpose needed.
                    d0, h0, w0 = img.shape
                    labels = mask_to_yolo_labels(mask, (d0, h0, w0))
                    
                    labels_list.append(labels)
                    shapes_list.append((d0, h0, w0))
                    segments_list.append([]) # Segments not used for NPZ
                except Exception as e:
                    print(f'{prefix}Warning: Could not process {img_file}: {e}')
                    # Append empty placeholders to maintain list alignment
                    labels_list.append(np.zeros((0, 7), dtype=np.float32))
                    shapes_list.append((0,0,0)) # Indicate error with zero shape
                    segments_list.append([])

                if (i + 1) % save_interval == 0 and i < len(files_to_process) - 1:
                    current_processed_files = list(processed_files) + files_to_process[:i+1]
                    part_cache_data = {
                        'labels': labels_list, 'shapes': shapes_list, 'segments': segments_list,
                        'img_files': current_processed_files, 'version': self.cache_version,
                    }
                    try:
                        temp_part_path = part_cache_path.with_suffix('.tmp')
                        torch.save(part_cache_data, temp_part_path)
                        temp_part_path.rename(part_cache_path)
                        pbar.set_description(f"{prefix}Saved intermediate cache progress")
                    except Exception as e:
                        print(f"\n{prefix}WARNING: Failed to save intermediate cache: {e}")

        elif self.file_format == 'nifti':
             # For NIfTI, we require external .txt label files.
            label_files = img2label_paths(files_to_process)
            with Pool(NUM_THREADS) as pool:
                pbar = tqdm(pool.imap(verify_image_label, zip(files_to_process, label_files, repeat(prefix))),
                            desc=f"{prefix}Processing nifti files for cache", total=len(files_to_process))
                for im_file, l, shape, segments, nm, nf, ne, nc, msg in pbar:
                    if im_file:
                        labels_list.append(l)
                        shapes_list.append(shape)
                        segments_list.append(segments)
                    if msg:
                        print(msg)
        
        # Finalize and save the complete cache
        final_processed_files = list(processed_files) + files_to_process
        final_cache_data = {
            'labels': labels_list,
            'shapes': np.array(shapes_list, dtype=np.float64),
            'segments': segments_list,
            'img_files': final_processed_files,
            'version': self.cache_version,
        }
        
        try:
            temp_final_path = cache_path.with_suffix('.tmp')
            torch.save(final_cache_data, temp_final_path)
            temp_final_path.rename(cache_path)
            print(f"{prefix}Successfully saved final cache to {cache_path}")
            if part_cache_path.is_file():
                part_cache_path.unlink() # Clean up partial file
        except Exception as e:
            print(f"{prefix}WARNING: Failed to save final cache: {e}")

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        """Caches dataset labels, verifies images and reads their shapes.
        See: verify_image_label function

        Args:
            path (pathlib.Path, optional): Path to write cache to. Defaults to Path('./labels.cache').
            prefix (str, optional): prefix for error messages. Defaults to ''.

        Returns:
            x (Dict): Dictionary containing the results of the image search.
        """
        # This function is now part of create_unified_cache and can be deprecated or removed.
        # For now, we leave it to avoid breaking other parts of the code that might call it,
        # but it is effectively unused by the new __init__ logic.
        pass

    def load_nifti(self, i):
        """Reads a nifti or npz file, converts it to a torch.tensor, and reshapes and resizes it for use in the YOLO3D model."""
        # loads 1 image from dataset index 'i'
        path = self.img_files[i]
        im, mask, affine = open_image(path, self.file_format)

        # reshape im from height, width, depth to depth, height, width to make it compatible with torch convolutions
        im = transpose_nifti_shape(im)

        d0, h0, w0 = im.size()

        # resize im to self.img_size
        im = change_nifti_size(im, self.img_size)

        return im, (d0, h0, w0), im.size()[1:], affine

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """Loads niftis and converts to torch tensor to be fed as input to the model.

        Args:
            index (int): dataset index of image to be read.

        Returns:
            img (torch.tensor): image data from loaded nifti, potentially augmented
            labels_out (torch.tensor): labels corresponding to loaded nifti, with augmentation accounted for
            self.img_files[index] (str): path to the loaded nifti
            shapes (Tuple[Tuple[float]]): Tuple containing Tuples of relative shape information for original image, resized image, and padding
        """
        # Load image
        img, (d0, h0, w0), (d, h, w), _ = self.load_nifti(self.indices[index])

        # Letterbox
        # shape = (self.img_size, self.img_size, self.img_size) # not adding rectangular training yet
        # img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment) # not implemented
        # ratio = (1, 1, 1) # no letterboxing so the shape doesn't change and the ratios are all 1
        pad = (0, 0, 0) # shape not changing so not padding any side
        shapes = (d0, h0, w0), ((d/d0, h/h0, w/w0), pad)

        labels = self.labels[self.indices[index]].copy()
        nl = len(labels)  # number of labels

        if self.augment:           
            # Label transformation is done to make certain augmentations more straightforward
            if labels.size:  # normalized zxydwh to pixel zxyzxy format
                labels[:, 1:] = zxydwhn2zxyzxy(labels[:, 1:], d, w, h, pad[0], pad[1], pad[2])                    

            # random zoom
            img, labels = random_zoom(img, labels, self.hyp['max_zoom'], self.hyp['min_zoom'], self.hyp['prob_zoom'])
        
            # transformation of labels back to standard format
            if nl:
                labels[:, 1:7] = zxyzxy2zxydwhn(labels[:, 1:7], d=img.shape[1], w=img.shape[3], h=img.shape[2], clip=True, eps=1E-3)
            
            # Albumentations
            
            # HSV color-space
            
            # Flip up-down
            
            # Flip left-right
            
            # Cutouts
            img, labels = tensor_cutout(img, labels, self.hyp['cutout_params'], self.hyp['prob_cutout'])
            # update after cutout
            nl = len(labels)  # number of labels
        
        labels_out = torch.zeros((nl, 8))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        return img, labels_out, self.img_files[self.indices[index]], shapes

    @staticmethod
    def collate_fn(batch):
        """Used to collate images to create the input batches"""
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def nifti_dataloader(path: str, imgsz: int, batch_size: int, stride: int, single_cls=False, hyp=None, augment=False, pad=0.0,
                     rank=-1, workers=16, prefix='', file_format='auto', cache_dir=None):
    """This is the dataloader used in the training process. Supports NIfTI and NPZ files.
    Args:
        file_format (str): 'auto', 'nifti', or 'npz'. If 'auto', detects format from file extensions. Default: 'auto'.
        cache_dir (str, optional): Path to a directory for caching dataset files. Defaults to None.
    """
    with torch_distributed_zero_first(rank):
        dataset = LoadNiftisAndLabels(path, imgsz, batch_size,
                                      augment=augment,
                                      hyp=hyp,
                                      # rect=rect,  # rectangular training
                                      single_cls=single_cls,
                                      stride=stride,
                                      pad=pad,
                                      prefix=prefix,
                                      file_format=file_format,
                                      cache_dir=cache_dir)

    batch_size = min(batch_size, len(dataset))
    # Memory optimization: Reduce workers for memory-constrained environments
    nw = min([os.cpu_count() // 2, batch_size if batch_size > 1 else 0, workers, 4])  # Reduced max workers to 4
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    
    # Use standard DataLoader when workers=0 to avoid multiprocessing issues
    if nw == 0:
        loader = torch.utils.data.DataLoader
        dataloader = loader(dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            sampler=sampler,
                            pin_memory=False,  # Disable for single process
                            collate_fn=LoadNiftisAndLabels.collate_fn)
    else:
        loader = InfiniteDataLoader
        dataloader = loader(dataset,
                            batch_size=batch_size,
                            num_workers=nw,
                            sampler=sampler,
                            pin_memory=True,  # Enable for faster GPU transfer
                            persistent_workers=True if nw > 0 else False,
                            prefetch_factor=2 if nw > 0 else None,
                            collate_fn=LoadNiftisAndLabels.collate_fn)
    
    return dataloader, dataset


def img2label_paths(img_paths):
    """Defines label paths as a function of the image paths.  Filters for .nii and .nii.gz files.

    Args:
        img_paths (List[str]): list of image file paths to convert to label file paths.

    Returns:
        List[str]: list of label file paths
    """
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    label_paths = []

    # to handle both compressed and uncompressed niftis
    for x in img_paths:
        if x.endswith('.nii.gz'):
            label_paths.append(sb.join(x.rsplit(sa, 1)).rsplit('.', 2)[0] + '.txt')
        elif x.endswith('.nii'):
            label_paths.append(sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt')

    return label_paths


def verify_image_label(args):
    """Verify one image-label pair.  Works for .nii and .nii.gz files.

    Args:
        args (Tuple[str]): contains the image path, label path, and error message prefix

    Returns:
        im_file (str): path to the image file
        l (List[float]): labels corresponding to the image file
        shape (List[int]): 3D shape of the image file
        segments: Alternate representation of image shape, currently not supported but necessary for compatibility with YOLOv5 code.
        nm (int): 1 if label missing, 0 if label found
        nf (int): 1 if label found, 0 if label not found
        ne (int): 1 if label empty, 0 if label not empty
        nc (int): 1 if label corrupted and Exception found, 0 if not
        msg (str): Message returned in the event an error occurs
    """
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = np.array(nib.load(im_file).dataobj)
        shape = (im.shape[2], im.shape[0], im.shape[1]) # need to transpose to account for depth reshaping that will happen to image tensors
        # assert call may need to be reworked for non-nifti data-types or if larger minimum sizes are required
        assert (shape[0] > 9) & (shape[1] > 99) & (shape[2] > 99), f'image size {shape} < 10x100x100 voxels'
        assert im_file.split('.')[-1].lower() in IMG_FORMATS or im_file.split('.')[-2].lower() in IMG_FORMATS, f'invalid image format {im_file}'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1 # label found
            with open(lb_file) as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                # segments aren't supported for simplicity
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert l.shape[1] == 7, f'labels require 7 columns, {l.shape[1]} columns detected'
                assert (l >= 0).all(), f'negative label values {l[l < 0]}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                l = np.unique(l, axis=0)  # remove duplicate rows
                if len(l) < nl:
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 7), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 7), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]
