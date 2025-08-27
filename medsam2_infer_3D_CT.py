from glob import glob
from tqdm import tqdm
import os
import sys
from os.path import join, basename
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import random
from pathlib import Path

from PIL import Image
import SimpleITK as sitk
import torch
import sam2  # Ensure sam2 package is imported to initialize Hydra
from sam2.build_sam import build_sam2_video_predictor_npz
from training.loss_fns import dice_loss,sigmoid_focal_loss
from skimage import measure, morphology
from scipy.spatial.distance import pdist, squareform
from typing import List, Optional, Tuple, Union,Dict,Any
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
import cv2

# Add MedYOLO to path for integration
sys.path.append('MedYOLO')

# MedYOLO imports for bounding box detection
try:
    from MedYOLO.val import run as medyolo_validate
    from MedYOLO.utils3D.datasets import nifti_dataloader
    from MedYOLO.models3D.model import attempt_load
    from MedYOLO.utils.torch_utils import select_device
    from MedYOLO.utils.general import check_dataset
    MEDYOLO_AVAILABLE = True
    print("MedYOLO integration enabled")
except ImportError as e:
    print(f"Warning: MedYOLO not available: {e}")
    print("Will use RECIST prompts instead of MedYOLO bounding boxes")
    MEDYOLO_AVAILABLE = False


torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2024)
np.random.seed(2024)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--checkpoint',
    type=str,
    default="FLARE_results/checkpoints/checkpoint.pt",
    help='checkpoint path',
)
parser.add_argument(
    '--cfg',
    type=str,
    default="configs/sam2.1_hiera_t512",
    help='model config',
)

parser.add_argument(
    '-i',
    '--imgs_path',
    type=str,
    default="data/validation_npz",
    help='imgs path',
)
#parser.add_argument(
    #'--gts_path',
    #default=None,
    #help='simulate prompts based on ground truth',
#)
parser.add_argument(
    '-o',
    '--pred_save_dir',
    type=str,
    default="FLARE_results",
    help='path to save segmentation results',
)

# MedYOLO integration arguments
parser.add_argument(
    '--use-medyolo',
    action='store_true',
    help='Use MedYOLO bounding box detection instead of RECIST prompts'
)
parser.add_argument(
    '--medyolo-data',
    type=str,
    default='MedYOLO/data/example.yaml',
    help='MedYOLO dataset YAML path'
)
parser.add_argument(
    '--medyolo-weights',
    type=str,
    default='./MedYOLO/runs/train/exp3/weights/last.pt',
    help='MedYOLO weights path'
)
parser.add_argument(
    '--medyolo-conf-thres',
    type=float,
    default=0.001,
    help='MedYOLO confidence threshold'
)
parser.add_argument(
    '--max-predictions',
    type=int,
    default=3,
    help='Maximum MedYOLO predictions to use as prompts'
)
parser.add_argument(
    '--prompt-strategy',
    type=str,
    default='auto',
    choices=['auto', 'medyolo', 'recist'],
    help='Prompt strategy: auto (use MedYOLO if available), medyolo, or recist'
)



args = parser.parse_args()
checkpoint = args.checkpoint
model_cfg = args.cfg
imgs_path = args.imgs_path
#gts_path = args.gts_path
pred_save_dir = args.pred_save_dir
os.makedirs(pred_save_dir, exist_ok=True)
#propagate_with_box = args.propagate_with_box

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def dice_multi_class(preds, targets):
    smooth = 1.0
    assert preds.shape == targets.shape
    labels = np.unique(targets)[1:]
    dices = []
    for label in labels:
        pred = preds == label
        target = targets == label
        intersection = (pred * target).sum()
        dices.append((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
    return np.mean(dices)

def calculate_dsc(pred_mask, gt_mask):
    """
    Calculate Dice Similarity Coefficient (DSC)
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
    
    Returns:
        float: DSC value between 0 and 1
    """
    smooth = 1e-6
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat)
    
    dsc = (2.0 * intersection + smooth) / (union + smooth)
    return dsc

def calculate_nsd(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate Normalized Surface Distance (NSD) with size-based optimization
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        spacing: Voxel spacing (z, y, x)
    
    Returns:
        float: NSD value (lower is better) or float('inf') if timeout
    """
    import time
    
    # Ensure masks are binary
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0
    
    # Check if masks are too large for efficient computation
    total_voxels = pred_mask.size
    if total_voxels > 1000 * 512 * 512:  # Increased threshold to handle larger volumes like 554x512x512
        print(f"  ⚠️  Very large volume detected ({total_voxels:,} voxels), skipping NSD")
        return float('inf')
    
    try:
        start_time = time.time()
        print(f"  Computing NSD for volume with {total_voxels:,} voxels...")
        
        # Calculate distance transforms (most time-consuming part)
        pred_dist = distance_transform_edt(~pred_binary, sampling=spacing)
        gt_dist = distance_transform_edt(~gt_binary, sampling=spacing)
        
        # Get surfaces
        pred_surface = pred_binary & (pred_dist <= 1.0)
        gt_surface = gt_binary & (gt_dist <= 1.0)
        
        if np.sum(pred_surface) == 0 or np.sum(gt_surface) == 0:
            return float('inf')  # Return infinity if no surface found
        
        # Calculate surface distances
        pred_to_gt_dist = distance_transform_edt(~gt_binary, sampling=spacing)[pred_surface]
        gt_to_pred_dist = distance_transform_edt(~pred_binary, sampling=spacing)[gt_surface]
        
        # Calculate NSD
        mean_pred_to_gt = np.mean(pred_to_gt_dist)
        mean_gt_to_pred = np.mean(gt_to_pred_dist)
        
        nsd = (mean_pred_to_gt + mean_gt_to_pred) / 2.0
        
        elapsed_time = time.time() - start_time
        print(f"  NSD calculation completed in {elapsed_time:.2f}s")
        
        return nsd
        
    except Exception as e:
        print(f"  ❌ Error in NSD calculation: {e}, returning inf")
        return float('inf')

def calculate_simplified_nsd(pred_binary, gt_binary, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate a simplified NSD using only boundary distances
    This is much faster but less accurate than the full NSD
    """
    try:
        # Get boundaries using morphological operations
        pred_boundary = pred_binary & ~ndimage.binary_erosion(pred_binary)
        gt_boundary = gt_binary & ~ndimage.binary_erosion(gt_binary)
        
        if np.sum(pred_boundary) == 0 or np.sum(gt_boundary) == 0:
            return float('inf')
        
        # Calculate distance from pred boundary to gt boundary
        pred_to_gt_dist = distance_transform_edt(~gt_binary, sampling=spacing)[pred_boundary]
        gt_to_pred_dist = distance_transform_edt(~pred_binary, sampling=spacing)[gt_boundary]
        
        # Calculate simplified NSD
        mean_pred_to_gt = np.mean(pred_to_gt_dist)
        mean_gt_to_pred = np.mean(gt_to_pred_dist)
        
        return (mean_pred_to_gt + mean_gt_to_pred) / 2.0
        
    except Exception as e:
        print(f"  ❌ Simplified NSD also failed: {e}")
        return float('inf')

def calculate_metrics(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate both DSC and NSD metrics
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        spacing: Voxel spacing (z, y, x)
    
    Returns:
        dict: Dictionary containing DSC and NSD values
    """
    dsc = calculate_dsc(pred_mask, gt_mask)
    nsd = calculate_nsd(pred_mask, gt_mask, spacing)
    
    return {
        'DSC': dsc,
        'NSD': nsd
    }

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array

def mask2D_to_bbox(gt2D, max_shift=20):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes

def mask3D_to_bbox(gt3D, max_shift=20):
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    D, H, W = gt3D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    z_min = max(0, z_min)
    z_max = min(D-1, z_max)
    boxes3d = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
    return boxes3d



def get_recist_marker_from_3d_mask(mask3d, spacing, direction, lesion_id=1):
    """
    Simulates a RECIST 2D marker from a 3D segmentation mask.
    It finds the slice with the largest lesion area and calculates both
    the longest diameter and perpendicular diameter in physical space.
    Returns 4 endpoints: (p1_long, p2_long, p1_perp, p2_perp, key_slice_id)
    """
    # 1. Find the key slice (largest area for the lesion)
    lesion_array = (mask3d == lesion_id).astype(np.uint8)
    if np.sum(lesion_array) == 0:
        return None, None, None, None, None # No lesion found

    area_per_slice = np.sum(lesion_array, axis=(1, 2))
    key_slice_id = np.argmax(area_per_slice)
    largest_2D_slice = lesion_array[key_slice_id, :, :]

    # 2. Find all points in the mask on that slice
    points_pix_yx = np.column_stack(np.where(largest_2D_slice > 0)) # gives (row, col) i.e. (y,x)
    if len(points_pix_yx) < 2:
        return None, None, None, None, key_slice_id

    # 3. Convert pixel indices to physical coordinates to find the true diameters
    direction_matrix = np.array(direction).reshape(3, 3)
    direction_2d = direction_matrix[:2, :2]
    spacing_2d = np.array(spacing)[:2]
    
    points_pix_xy = np.flip(points_pix_yx, axis=1) # now (x, y)
    phys_points = (points_pix_xy * spacing_2d) @ direction_2d.T
    
    # 4. Find the longest diameter (RECIST measurement)
    dist_matrix = squareform(pdist(phys_points))
    max_diam_indices = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    
    # 5. Find the perpendicular diameter (perpendicular to the longest diameter)
    p1_long_phys = phys_points[max_diam_indices[0]]
    p2_long_phys = phys_points[max_diam_indices[1]]
    
    # Vector of longest diameter
    long_vector = p2_long_phys - p1_long_phys
    long_vector = long_vector / np.linalg.norm(long_vector)
    
    # Perpendicular vector
    perp_vector = np.array([-long_vector[1], long_vector[0]])
    
    # Project all points onto the perpendicular line and find the two furthest apart
    perp_distances = np.dot(phys_points - p1_long_phys, perp_vector)
    perp_min_idx = np.argmin(perp_distances)
    perp_max_idx = np.argmax(perp_distances)
    
    # 6. Get the corresponding pixel coordinates of all 4 end points
    p1_long = points_pix_xy[max_diam_indices[0]]  # (x, y) - longest diameter endpoint 1
    p2_long = points_pix_xy[max_diam_indices[1]]  # (x, y) - longest diameter endpoint 2
    p1_perp = points_pix_xy[perp_min_idx]         # (x, y) - perpendicular diameter endpoint 1
    p2_perp = points_pix_xy[perp_max_idx]         # (x, y) - perpendicular diameter endpoint 2
    
    return p1_long, p2_long, p1_perp, p2_perp, key_slice_id




def visualize_and_handle_output(img3d, gts3d, segs3d, spacing, origin, direction, out_path=None,dsc=None,nsd=None,time_taken=None):
    """
    Generates a simple visualization with the RECIST marker (from ground truth) and segmentation.
    - If out_path is provided, saves the figure to the path.
    - If out_path is None, displays the figure interactively.
    """
    p1_long, p2_long, p1_perp, p2_perp, slice_idx = get_recist_marker_from_3d_mask(gts3d, spacing, direction)
    
    if slice_idx is None:
        if out_path: # Only print for the saved files
            print(f"Skipping visualization for {os.path.basename(out_path)}: no segmentation found.")
        return

    orig = img3d[slice_idx]
    pred2d = segs3d[slice_idx]

    # Simple visualization with two panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Panel 1: Original image with 4 RECIST endpoints
    axes[0].imshow(orig, cmap='gray')
    if p1_long is not None and p2_long is not None and p1_perp is not None and p2_perp is not None:
        # Draw longest diameter
        axes[0].plot([p1_long[0], p2_long[0]], [p1_long[1], p2_long[1]], 'r-', linewidth=2, label='Longest Diameter')
        axes[0].scatter([p1_long[0], p2_long[0]], [p1_long[1], p2_long[1]], c='red', s=50, zorder=5)
        
        # Draw perpendicular diameter
        axes[0].plot([p1_perp[0], p2_perp[0]], [p1_perp[1], p2_perp[1]], 'b-', linewidth=2, label='Perpendicular Diameter')
        axes[0].scatter([p1_perp[0], p2_perp[0]], [p1_perp[1], p2_perp[1]], c='blue', s=50, zorder=5)
        
    axes[0].set_title(f'Slice #{slice_idx} with 4 RECIST Endpoints')
    axes[0].axis('off')
    axes[0].legend()
    
    # Panel 2: 3D segmentation overlay
    axes[1].imshow(orig, cmap='gray')
    show_mask(pred2d, axes[1], mask_color=np.array([0,1,0]), alpha=0.5) # Green mask
    axes[1].set_title('3D Segmentation Overlay')
    metrics_text = []
    if dsc is not None:
        metrics_text.append(f"DSC: {dsc:.3f}")
    if nsd is not None:
        metrics_text.append(f"NSD: {nsd:.3f}")
    if time_taken is not None:
        metrics_text.append(f"Time: {time_taken:.2f}s")
    text_str = "\n".join(metrics_text)
    
    # place inside axes coords (x=0.01,y=0.99), anchored top-left
    axes[1].text(
        0.01, 0.99, text_str,
        transform=axes[1].transAxes,
        fontsize=12,
        color='white',
        verticalalignment='top',
        bbox=dict(facecolor='black', alpha=0.6, pad=5)
    )
    axes[1].axis('off')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def visualize_medyolo_results(img3d, segs3d, medyolo_preds, slice_idx=None, out_path=None, prompts=None, labels=None,
                              dsc=None, nsd=None, num_preds=None,time_taken=None):
    """Visualize MedYOLO predictions and segmentation results"""
    print(f"=== visualize_medyolo_results FUNCTION CALLED ===")
    print(f"DEBUG: visualize_medyolo_results called with out_path={out_path}")
    print(f"DEBUG: segs3d.max() = {segs3d.max()}, segs3d.sum() = {segs3d.sum()}")
    
    # Force visualization even if segmentation is empty to debug the issue
    if segs3d.max() == 0:
        print(f"WARNING: Empty segmentation detected, but creating visualization anyway for debugging")
        # Don't return early - continue to create visualization

    # Use middle slice if not specified
    if slice_idx is None:
        slice_idx = img3d.shape[0] // 2

    orig = img3d[slice_idx]
    pred2d = segs3d[slice_idx]
    
    # Resize original image to 512x512 for consistent visualization with unscaled coordinates
    from skimage.transform import resize
    orig_512 = resize(orig, (512, 512), order=1, preserve_range=True).astype(np.uint8)
    pred2d_512 = resize(pred2d, (512, 512), order=0, preserve_range=True).astype(np.uint8)

    # Simple visualization with two panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Panel 1: Original image with MedYOLO bounding boxes AND corner points
    axes[0].imshow(orig_512, cmap='gray')
    
    # Draw MedYOLO bounding boxes and corner points
    print(f"DEBUG: medyolo_preds type: {type(medyolo_preds)}, length: {len(medyolo_preds) if medyolo_preds else 0}")
    if medyolo_preds and len(medyolo_preds) > 0:
        print(f"DEBUG: First prediction keys: {list(medyolo_preds[0].keys()) if medyolo_preds else 'None'}")
        
        # Show the actual prompts that were sent to MedSAM2 (if available)
        if prompts is not None and labels is not None:
            print(f"DEBUG: Showing actual MedSAM2 prompts: {len(prompts)} points")
            points_per_pred = 4
            num_predictions = len(prompts) // points_per_pred
            
            for i in range(num_predictions):
                # Get 4 points for this prediction
                start_idx = i * points_per_pred
                end_idx = start_idx + points_per_pred
                pred_points = prompts[start_idx:end_idx]
                pred_labels = labels[start_idx:end_idx]
                
                # Use prompts directly in 512x512 space (no scaling needed)
                corner_points = [(int(point[0]), int(point[1])) for point in pred_points]
                
                # Plot corner points with just 2 colors (red and blue)
                point_colors = ['red', 'blue', 'red', 'blue']  # Alternate colors
                for j, (cx, cy) in enumerate(corner_points):
                    axes[0].scatter(cx, cy, c=point_colors[j], s=150, marker='o', zorder=10)
                
                # Draw lines connecting the actual points used for segmentation
                # Longest diameter: Top-left to Bottom-right
                axes[0].plot([corner_points[0][0], corner_points[3][0]], 
                           [corner_points[0][1], corner_points[3][1]], 
                           'r-', linewidth=2, alpha=0.7, label='MedSAM2 diameter' if i == 0 else "")
                
                # Perpendicular diameter: Top-right to Bottom-left  
                axes[0].plot([corner_points[1][0], corner_points[2][0]], 
                           [corner_points[1][1], corner_points[2][1]], 
                           'b-', linewidth=2, alpha=0.7, label='MedSAM2 diameter' if i == 0 else "")
                
                # Add object label
                axes[0].text(corner_points[0][0], corner_points[0][1]-10, f'Obj{i+1}', 
                           color='red', fontsize=10, weight='bold')
        
        # Also show original bounding boxes for reference (no scaling needed)
        for i, pred in enumerate(medyolo_preds):
            bbox = pred['bbox']  # [z1, x1, y1, z2, x2, y2] 
            conf = pred.get('confidence', 0)
            
            z1, x1, y1, z2, x2, y2 = bbox
            
            # Use coordinates directly in 512x512x512 space (no scaling)
            z1_scaled = int(z1)
            x1_scaled = int(x1)
            y1_scaled = int(y1)
            z2_scaled = int(z2)
            x2_scaled = int(x2)
            y2_scaled = int(y2)
            
            # Check if this prediction is in the current slice
            if z1_scaled <= slice_idx <= z2_scaled:
                # Draw bounding box (dashed for reference)
                rect = plt.Rectangle((x1_scaled, y1_scaled), x2_scaled-x1_scaled, y2_scaled-y1_scaled, 
                                   linewidth=2, edgecolor='orange', facecolor='none', alpha=0.6, linestyle='--')
                axes[0].add_patch(rect)
                
                # Add confidence text
                axes[0].text(x1_scaled, y1_scaled-5, f'MedYOLO{i+1}: {conf:.3f}', 
                           color='orange', fontsize=8, weight='bold')
        
        axes[0].set_title(f'Slice #{slice_idx} with MedYOLO Predictions')
        # Only show legend for first object to avoid clutter
        if len(medyolo_preds) > 0:
            handles, labels_legend = axes[0].get_legend_handles_labels()
            # Keep only unique labels
            unique_labels = []
            unique_handles = []
            for handle, label in zip(handles, labels_legend):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            axes[0].legend(unique_handles, unique_labels, loc='upper right')
    else:
        axes[0].set_title(f'Slice #{slice_idx} - No MedYOLO Predictions')
    
    axes[0].axis('off')
    
    # Panel 2: 3D segmentation overlay
    axes[1].imshow(orig_512, cmap='gray') # Use resized image for overlay
    if pred2d_512.max() > 0:
        show_mask(pred2d_512, axes[1], mask_color=np.array([0,1,0]), alpha=0.5) # Green mask
        axes[1].set_title('MedYOLO + MedSAM2 Segmentation')
    else:
        axes[1].set_title('MedYOLO + MedSAM2 Segmentation (No Segmentation)')
    # now add text box in upper-left
    metrics_text = []
    if dsc is not None:
        metrics_text.append(f"DSC: {dsc:.3f}")
    if nsd is not None:
        metrics_text.append(f"NSD: {nsd:.3f}")
    if num_preds is not None:
        metrics_text.append(f"#preds: {num_preds}")
    if time_taken is not None:
        metrics_text.append(f"Time: {time_taken:.2f}s")
    text_str = "\n".join(metrics_text)
    
    # place inside axes coords (x=0.01,y=0.99), anchored top-left
    axes[1].text(
        0.01, 0.99, text_str,
        transform=axes[1].transAxes,
        fontsize=12,
        color='white',
        verticalalignment='top',
        bbox=dict(facecolor='black', alpha=0.6, pad=5)
    )
    axes[1].axis('off')

    plt.tight_layout()
    if out_path:
        print(f"DEBUG: Saving visualization to {out_path}")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"DEBUG: Visualization saved successfully")
    else:
        plt.show()





# MedYOLO Integration Functions
def run_medyolo_validation_consistent(data_yaml, weights_path, conf_thres=0.001, max_retries=3):
    """Run MedYOLO validation with consistency improvements"""
    print("Running MedYOLO validation with consistency improvements...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Import MedYOLO validation function directly
    import sys
    sys.path.append('MedYOLO')
    from val import run as run_medyolo_val
    
    # Try multiple times to get consistent results
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}")
            
            predictions = run_medyolo_val(
                data=data_yaml,
                weights=weights_path,
                conf_thres=conf_thres,
                return_predictions=True,
                save_visualizations=False,
                verbose=False,
                plots=False,
                save_txt=False
            )
            
            # Check if we got reasonable predictions
            total_predictions = sum(len(data['predictions']) for data in predictions.values())
            print(f"  Got {total_predictions} total predictions across {len(predictions)} images")
            
            if total_predictions > 0:
                print(f"  MedYOLO validation completed successfully on attempt {attempt + 1}")
                return predictions
            else:
                print(f"  No predictions found, retrying...")
                
        except Exception as e:
            print(f"  Error in attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                raise e
    
    print(f"  Warning: No predictions found after {max_retries} attempts")
    return {}

def load_medyolo_predictions(val_dir):
    """Load MedYOLO predictions from saved text files"""
    predictions = {}
    labels_dir = Path(val_dir) / 'labels'
    
    if not labels_dir.exists():
        print(f"Warning: Labels directory {labels_dir} not found")
        return predictions
    
    for txt_file in labels_dir.glob('*.txt'):
        filename = txt_file.stem
        predictions[filename] = []
        
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:  # class, z, x, y, d, w, h, [conf]
                    cls = int(parts[0])
                    z, x, y, d, w, h = map(float, parts[1:7])
                    conf = float(parts[7]) if len(parts) > 7 else 1.0
                    
                    predictions[filename].append({
                        'class': cls,
                        'z': z, 'x': x, 'y': y, 'd': d, 'w': w, 'h': h,
                        'confidence': conf,
                        'bbox_3d': [z, x, y, d, w, h]
                    })
    
    return predictions

def bbox_3d_to_2d_bbox(bbox_3d, img_shape):
    """Convert 3D bounding box to 2D bounding box for MedSAM2"""
    # MedYOLO format: [z1, x1, y1, z2, x2, y2] (corner coordinates)
    z1, x1, y1, z2, x2, y2 = bbox_3d
    D, H, W = img_shape
    
    # MedYOLO is already using 512x512x512 coordinates, same as MedSAM2
    # No scaling needed - use coordinates directly
    
    z1_scaled = z1
    x1_scaled = x1
    y1_scaled = y1
    z2_scaled = z2
    x2_scaled = x2
    y2_scaled = y2
    
    # Convert to pixel coordinates in 512x512x512 space
    z_center = int(z1_scaled)
    x_min = int(x1_scaled)
    y_min = int(y1_scaled)
    x_max = int(x2_scaled)
    y_max = int(y2_scaled)
    
    # Ensure coordinates are within bounds
    x_min = max(0, min(W-1, x_min))
    y_min = max(0, min(H-1, y_min))
    x_max = max(0, min(W-1, x_max))
    y_max = max(0, min(H-1, y_max))
    
    # Return as 2D bounding box [x_min, y_min, x_max, y_max]
    return [x_min, y_min, x_max, y_max]

def bbox_3d_to_corner_prompts(bbox_3d, img_shape):
    """Convert 3D bounding box to 4 corner point prompts for MedSAM2"""
    # MedYOLO format: [z1, x1, y1, z2, x2, y2] (corner coordinates)
    z1, x1, y1, z2, x2, y2 = bbox_3d
    D, H, W = img_shape
    
    print(f"  Raw bbox: [{z1:.3f}, {x1:.3f}, {y1:.3f}, {z2:.3f}, {x2:.3f}, {y2:.3f}]")
    print(f"  Image shape: D={D}, H={H}, W={W}")
    
    # MedYOLO is already using 512x512x512 coordinates, same as MedSAM2
    # No scaling needed - use coordinates directly
    print(f"  Using MedYOLO coordinates directly (512x512x512 space)")
    
    z1_scaled = z1
    x1_scaled = x1
    y1_scaled = y1
    z2_scaled = z2
    x2_scaled = x2
    y2_scaled = y2
    
    # Convert to integers
    z1_px = int(z1_scaled)
    x1_px = int(x1_scaled)
    y1_px = int(y1_scaled)
    z2_px = int(z2_scaled)
    x2_px = int(x2_scaled)
    y2_px = int(y2_scaled)
    
    # Ensure coordinates are within bounds
    z1_px = max(0, min(D-1, z1_px))
    x1_px = max(0, min(W-1, x1_px))
    y1_px = max(0, min(H-1, y1_px))
    z2_px = max(0, min(D-1, z2_px))
    x2_px = max(0, min(W-1, x2_px))
    y2_px = max(0, min(H-1, y2_px))
    
    # Calculate dimensions
    width_px = x2_px - x1_px
    height_px = y2_px - y1_px
    
    # Ensure minimum dimensions
    width_px = max(1, width_px)
    height_px = max(1, height_px)
    
    print(f"  Corner bbox: [{z1_px}, {x1_px}, {y1_px}, {z2_px}, {x2_px}, {y2_px}]")
    
    # Use the 4 corners of the bounding box as prompts
    # Top-left, top-right, bottom-left, bottom-right
    corner_points = [
        x1_px, y1_px,  # Top-left
        x2_px, y1_px,  # Top-right  
        x1_px, y2_px,  # Bottom-left
        x2_px, y2_px   # Bottom-right
    ]
    
    print(f"  Corner points: TL=({x1_px},{y1_px}), TR=({x2_px},{y1_px}), BL=({x1_px},{y2_px}), BR=({x2_px},{y2_px})")
    
    # Return as 4-point prompt: [x1, y1, x2, y1, x1, y2, x2, y2]
    return corner_points

def get_medyolo_prompts(npz_fname, medyolo_predictions, img_shape, max_predictions=6, use_bbox=False):
    """Get MedYOLO prompts for a specific NPZ file"""
    base_name = os.path.splitext(npz_fname)[0]
    
    if base_name not in medyolo_predictions:
        print(f"No MedYOLO predictions found for {base_name}")
        return None, None, None
    
    data = medyolo_predictions[base_name]
    predictions = data['predictions']  # List of prediction dictionaries
    
    if not predictions:
        print(f"No valid MedYOLO predictions for {base_name}")
        return None, None, None
    
    # Predictions are already sorted by confidence and limited to top 6 from validation
    top_preds = predictions[:max_predictions]
    
    if use_bbox:
        # Convert to bounding box prompts
        all_bboxes = []
        all_labels = []
        
        for pred in top_preds:
            bbox = pred['bbox']  # [z1, x1, y1, z2, x2, y2]
            bbox_2d = bbox_3d_to_2d_bbox(bbox, img_shape)  # Use actual image shape
            all_bboxes.append(bbox_2d)
            all_labels.append(1)  # Positive box
        
        if not all_bboxes:
            print(f"No valid bounding box prompts generated for {base_name}")
            return None, None, None
        
        bboxes = np.array(all_bboxes)
        labels = np.array(all_labels, dtype=np.int32)
        
        print(f"{base_name}: Using {len(bboxes)} bounding box prompts from {len(top_preds)} MedYOLO predictions")
        return bboxes, labels, top_preds
    else:
        # Convert to RECIST-style 4-point prompts
        all_points = []
        all_labels = []
        
        for pred in top_preds:
            # Debug: print available keys
            print(f"  Available keys in prediction: {list(pred.keys())}")
            
            # MedYOLO predictions are already in [z1, x1, y1, z2, x2, y2] format (corner coordinates)
            # from the validation function
            bbox_corners = pred['bbox']  # [z1, x1, y1, z2, x2, y2]
            print(f"  MedYOLO prediction: bbox={bbox_corners}, conf={pred.get('confidence', 'N/A')}")
            
            # MedYOLO coordinates are already in 512x512x512 space, same as MedSAM2
            # No scaling needed - convert directly to corner points
            corner_points = bbox_3d_to_corner_prompts(bbox_corners, img_shape)
            all_points.extend(corner_points)
            all_labels.extend([1, 1, 1, 1])  # 4 positive points per prediction
        
        if not all_points:
            print(f"No valid RECIST prompts generated for {base_name}")
            return None, None, None
        
        points = np.array(all_points).reshape(-1, 2)
        labels = np.array(all_labels, dtype=np.int32)
        
        print(f"{base_name}: Using {len(points)} RECIST-style points from {len(top_preds)} MedYOLO predictions")
        return points, labels, top_preds


npz_fnames = sorted(os.listdir(imgs_path))
npz_fnames = [i for i in npz_fnames if i.endswith('.npz')]
npz_fnames = [i for i in npz_fnames if not i.startswith('._')]
print(f'Processing {len(npz_fnames)} npz files')

# Determine which strategies to run
run_medyolo = MEDYOLO_AVAILABLE and args.use_medyolo
run_recist = True  # Always run RECIST for comparison

print(f"Running strategies:")
print(f"  - MedYOLO: {'Yes' if run_medyolo else 'No'}")
print(f"  - RECIST: {'Yes' if run_recist else 'No'}")

# Initialize MedYOLO predictions if needed
medyolo_predictions = {}
if run_medyolo:
    print("Running MedYOLO validation to get bounding box predictions...")
    medyolo_predictions = run_medyolo_validation_consistent(
        args.medyolo_data, 
        args.medyolo_weights, 
        args.medyolo_conf_thres
    )
    
    if medyolo_predictions:
        print(f"Got MedYOLO predictions for {len(medyolo_predictions)} files")
    else:
        print("MedYOLO validation failed, will skip MedYOLO strategy")
        run_medyolo = False

# Initialize results storage for both strategies
seg_info_medyolo = {"npz_name":[], "key_slice_index":[], "DSC":[], "NSD":[], "strategy":[], "num_predictions":[]}
seg_info_recist = {"npz_name":[], "key_slice_index":[], "DSC":[], "NSD":[], "strategy":[], "num_predictions":[]}

# initialized predictor
predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

# Before the main loop
data_cache = {}
import time
total_start_time = time.time()
case_times = []

for npz_fname in tqdm(npz_fnames):
    if not npz_fname.endswith(".npz"):
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing: {npz_fname}")
    print(f"{'='*60}")
    
    data = np.load(os.path.join(imgs_path, npz_fname))
    img3d = data["imgs"]    # shape (Z,H,W), uint8 or float
    spacing = tuple(data["spacing"].tolist()) 
    direction = data["direction"]  # shape (9,)
    origin  = tuple(data["origin"].tolist()) 
    H, W = img3d.shape[1:]

    # 1) normalize raw volume once
    vol = img3d.astype(np.float32)
    # if needed, clip to known window; otherwise just scale full min→max:
    vol = ((vol - vol.min())/(vol.max()-vol.min()) * 255.0).astype(np.uint8)
    gts3d = data["gts"] if "gts" in data and data["gts"].max() > 0 else np.zeros_like(img3d, dtype=np.uint8)

    # 4) prepare predictor input (shared for both strategies)
    # MedSAM2 expects 512x512, so resize to 512x512x512
    vol_rgb   = resize_grayscale_to_rgb_and_resize(vol, 512) / 255.0
    img_tensor = torch.from_numpy(vol_rgb).float().to(device)
    mean = torch.tensor([0.485,0.456,0.406], dtype=torch.float32)[:,None,None].to(device)
    std  = torch.tensor([0.229,0.224,0.225], dtype=torch.float32)[:,None,None].to(device)
    img_tensor.sub_(mean).div_(std)

    base = os.path.splitext(npz_fname)[0]
    
    # Determine the slice to use for both strategies
    # Use RECIST slice (slice with largest lesion area) if available, otherwise middle slice
    if "gts" in data and data["gts"].max() > 0:
        # Get RECIST slice (slice with largest lesion area)
        p1_long, p2_long, p1_perp, p2_perp, zmid = get_recist_marker_from_3d_mask(
            mask3d=gts3d,
            spacing=spacing,
            direction=direction,
            lesion_id=1
        )
        if zmid is None:
            zmid = img3d.shape[0] // 2  # Fallback to middle slice
    else:
        zmid = img3d.shape[0] // 2  # Use middle slice if no ground truth
    
    print(f"Using slice {zmid} for both MedYOLO and RECIST strategies")
    
    # ===== STRATEGY 1: MEDYOLO =====
    if run_medyolo:
        print(f"\n--- STRATEGY 1: MEDYOLO ---")
        start_time = time.time()
        
        # Get MedYOLO prompts (always use points)
        # Use the actual image shape that MedSAM2 will use (512x512x512)
        medyolo_img_shape = (img3d.shape[0], 512, 512) 
        prompts, labels, medyolo_preds = get_medyolo_prompts(
            npz_fname, medyolo_predictions, medyolo_img_shape, args.max_predictions, use_bbox=False
        )
        
        print(f"DEBUG: prompts shape: {prompts.shape if prompts is not None else 'None'}")
        print(f"DEBUG: labels shape: {labels.shape if labels is not None else 'None'}")
        print(f"DEBUG: medyolo_preds: {len(medyolo_preds) if medyolo_preds is not None else 'None'}")
        
        if prompts is not None:
            print(f"DEBUG: First few prompts: {prompts[:8] if len(prompts) >= 8 else prompts}")
            print(f"DEBUG: First few labels: {labels[:8] if len(labels) >= 8 else labels}")
            
            strategy_name = f'MedYOLO_{len(medyolo_preds) if medyolo_preds else 0}_predictions_points'
            
            # Run MedSAM2 segmentation with MedYOLO prompts
            autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16):
                state = predictor.init_state(
                    img_tensor,
                    video_height=H,
                    video_width=W
                )
                
                # Use MedYOLO predictions as RECIST-style prompts
                print(f"Using {len(medyolo_preds) if medyolo_preds else 0} MedYOLO predictions as RECIST-style prompts")
                
                # Process each prediction as a separate object with RECIST-style 4 points
                segs_3D_medyolo = np.zeros_like(vol, dtype=np.uint8)
                
                # Group points by prediction (4 points per prediction)
                points_per_pred = 4
                num_predictions = len(prompts) // points_per_pred
                
                print(f"Processing {num_predictions} predictions with {points_per_pred} points each")
                
                for i in range(num_predictions):
                    obj_id = i + 1  # Each prediction gets a unique object ID
                    
                    # Get 4 points for this prediction
                    start_idx = i * points_per_pred
                    end_idx = start_idx + points_per_pred
                    pred_points = prompts[start_idx:end_idx]
                    pred_labels = labels[start_idx:end_idx]
                    
                    # Validate RECIST points
                    x_coords = pred_points[:, 0]
                    y_coords = pred_points[:, 1]
                    x_range = x_coords.max() - x_coords.min()
                    y_range = y_coords.max() - y_coords.min()
                    
                    print(f"Object {obj_id}: Points {pred_points}, Labels {pred_labels}")
                    print(f"  Point ranges: X={x_range}, Y={y_range}")
                    
                    # Check if points are too close together (relaxed threshold for MedYOLO)
                    if x_range < 2 or y_range < 2:
                        print(f"  WARNING: Points too close together! Skipping object {obj_id}")
                        continue
                    
                    print(f"  Adding object {obj_id} with {len(pred_points)} points")
                    _, _, logit = predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=zmid,
                        obj_id=obj_id,
                        points=pred_points,  # 4 points as 2D array
                        labels=pred_labels
                    )
                   
                # Propagate through video for all objects
                total_masks = 0
                frame_count = 0
                
                print(f"Starting mask propagation through {img3d.shape[0]} frames...")
                
                # Use the propagate_in_video method to get masks for all frames
                try:
                    total_objects_processed = 0
                    total_masks_added = 0
                    
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state, reverse=True):
                        print(f"  Frame {out_frame_idx}: Processing {len(out_mask_logits)} objects")
                        # Process ALL objects, not just the first one
                        for obj_idx, mask_logits in enumerate(out_mask_logits):
                            if mask_logits is not None:
                                binary_mask = (mask_logits > 0.0).cpu().numpy()[0]
                                if binary_mask.sum() > 0:  # Only add if mask has content
                                    segs_3D_medyolo[out_frame_idx, binary_mask] = 1
                                    total_masks_added += 1
                                    print(f"    Object {obj_idx+1}: mask sum = {binary_mask.sum()}")
                            total_objects_processed += 1
                    
                    print(f"  Total objects processed: {total_objects_processed}")
                    print(f"  Total masks added: {total_masks_added}")
                    predictor.reset_state(state)
                except Exception as e:
                    print(f"  Error during propagation: {e}")
                    # Fallback: try to get masks from the state directly
                    print("  Attempting fallback method...")
                    output_dict = state.get("output_dict", {})
                    for frame_idx in range(img3d.shape[0]):
                        try:
                            # Check both cond and non_cond frame outputs
                            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                                if frame_idx in output_dict.get(storage_key, {}):
                                    current_out = output_dict[storage_key][frame_idx]
                                    pred_masks = current_out.get("pred_masks")
                                    if pred_masks is not None and len(pred_masks) > 0:
                                        frame_mask = np.zeros((H, W), dtype=np.uint8)
                                        for mask in pred_masks:
                                            if mask is not None:
                                                if torch.is_tensor(mask):
                                                    mask = mask.cpu().numpy()
                                                
                                                # Apply threshold to convert logits to binary mask
                                                binary_mask = (mask > 0.0).astype(np.uint8)
                                                
                                                if binary_mask.sum() > 0:
                                                    frame_mask = np.logical_or(frame_mask, binary_mask).astype(np.uint8)
                                        # Resize from 512x512 to original dimensions
                                        if frame_mask.shape != (H, W):
                                            from skimage.transform import resize
                                            frame_mask = resize(frame_mask, (H, W), order=0, preserve_range=True).astype(np.uint8)
                                        segs_3D_medyolo[frame_idx] = frame_mask
                                        break
                        except Exception as frame_error:
                            print(f"    Error processing frame {frame_idx}: {frame_error}")
                            continue
                
                print(f"Processed {frame_count} frames")
                print(f"Generated {total_masks} masks across all frames")
                print(f"Final segmentation volume shape: {segs_3D_medyolo.shape}")
                print(f"Segmentation volume max value: {segs_3D_medyolo.max()}")
                print(f"Segmentation volume sum: {segs_3D_medyolo.sum()}")
                
                predictor.reset_state(state)
            
            # Keep only largest connected component
            if segs_3D_medyolo.max() > 0:
                segs_3D_medyolo = getLargestCC(segs_3D_medyolo).astype(np.uint8)
            
            # Save MedYOLO results
            itk_img = sitk.GetImageFromArray(img3d)
            itk_mask_medyolo = sitk.GetImageFromArray(segs_3D_medyolo)
            
            # Set metadata
            itk_img.SetSpacing(spacing)
            itk_img.SetDirection(direction.flatten())
            itk_img.SetOrigin(origin)
            
            # Save files with strategy suffix
            img_path = os.path.join(pred_save_dir, f"{base}_img.nii.gz")
            mask_path_medyolo = os.path.join(pred_save_dir, f"{base}_mask_medyolo.nii.gz")
            
            sitk.WriteImage(itk_img, img_path)
            sitk.WriteImage(itk_mask_medyolo, mask_path_medyolo)
            
            
        
            # Calculate metrics for MedYOLO
            dsc_medyolo = float('nan')
            nsd_medyolo = float('nan')
            if "gts" in data and data["gts"].max() > 0:
                metrics = calculate_metrics(segs_3D_medyolo, gts3d, spacing)
                dsc_medyolo = metrics['DSC']
                nsd_medyolo = metrics['NSD']
                print(f"MedYOLO Metrics: DSC = {dsc_medyolo:.4f}, NSD = {nsd_medyolo:.4f}")
            
            # Store MedYOLO results
            seg_info_medyolo["npz_name"].append(f"{base}_mask_medyolo.nii.gz")
            seg_info_medyolo["key_slice_index"].append(int(zmid))
            seg_info_medyolo["DSC"].append(dsc_medyolo)
            seg_info_medyolo["NSD"].append(nsd_medyolo)
            seg_info_medyolo["strategy"].append(strategy_name)
            seg_info_medyolo["num_predictions"].append(len(medyolo_preds) if medyolo_preds else 0)
            
            medyolo_time = time.time() - start_time
            print(f"MedYOLO completed in: {medyolo_time:.2f} seconds")

            # Create MedYOLO visualization
            viz_path = os.path.join(pred_save_dir, f"{base}_medyolo_visualization.png")
            print(f"DEBUG: Creating MedYOLO visualization at {viz_path}")
            print(f"DEBUG: segs_3D_medyolo.max() = {segs_3D_medyolo.max()}")
            print(f"DEBUG: segs_3D_medyolo.sum() = {segs_3D_medyolo.sum()}")
            print(f"DEBUG: medyolo_preds = {medyolo_preds}")
            visualize_medyolo_results(img3d, segs_3D_medyolo, medyolo_preds, zmid, viz_path, prompts, labels,
                dsc=dsc_medyolo,nsd=nsd_medyolo,num_preds=len(medyolo_preds),time_taken=medyolo_time)
            print(f"DEBUG: Visualization function completed")
        else:
            print("MedYOLO prompts failed, skipping MedYOLO strategy")
    
    # ===== STRATEGY 2: RECIST =====
    if run_recist:
        print(f"\n--- STRATEGY 2: RECIST ---")
        start_time = time.time()
        
        # Get RECIST prompts
        if "gts" in data and data["gts"].max() > 0:
            # Use the same slice that was determined at the beginning
            # Re-extract RECIST endpoints for the determined slice
            p1_long, p2_long, p1_perp, p2_perp, _ = get_recist_marker_from_3d_mask(
                mask3d=gts3d,
                spacing=spacing,
                direction=direction,
                lesion_id=1
            )

            if p1_long is not None and p2_long is not None and p1_perp is not None and p2_perp is not None:
                # Create point prompt with all 4 endpoints
                point_prompt = [
                    int(p1_long[0]), int(p1_long[1]),  # Longest diameter endpoint 1
                    int(p2_long[0]), int(p2_long[1]),  # Longest diameter endpoint 2
                    int(p1_perp[0]), int(p1_perp[1]),  # Perpendicular diameter endpoint 1
                    int(p2_perp[0]), int(p2_perp[1])   # Perpendicular diameter endpoint 2
                ]
                labels_prompt = [1, 1, 1, 1]  # All positive points
                strategy_name = 'RECIST_2D_Marker'
                print(f"RECIST 4 endpoints: Long=({p1_long}, {p2_long}), Perp=({p1_perp}, {p2_perp})")
            else:
                print(f"RECIST extraction failed for {npz_fname}. Using center point.")
                point_prompt = [W // 2, H // 2]
                labels_prompt = [1]
                strategy_name = 'Center_Point'
        else:
            print(f"No GT found for {npz_fname}. Using center point of image.")
            point_prompt = [W // 2, H // 2]
            labels_prompt = [1]
            strategy_name = 'Center_Point'
        
        # Run MedSAM2 segmentation with RECIST prompts
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with torch.inference_mode(), torch.autocast(autocast_device, dtype=torch.bfloat16):
            state = predictor.init_state(
                img_tensor,
                video_height=H,
                video_width=W
            )
            
            points = np.array(point_prompt).reshape(-1, 2)
            labels = np.array(labels_prompt, dtype=np.int32)
            
            # Use RECIST endpoints as point prompts
            _, _, logit = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=zmid,
                obj_id=1,
                points=points,
                labels=labels
            )
            
            # Propagate through video
            segs_3D_recist = np.zeros_like(vol, dtype=np.uint8)
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state, reverse=True):
                segs_3D_recist[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
            predictor.reset_state(state)
        
        # Keep only largest connected component
        if segs_3D_recist.max() > 0:
            segs_3D_recist = getLargestCC(segs_3D_recist).astype(np.uint8)
        
        # Save RECIST results
        itk_img = sitk.GetImageFromArray(img3d)
        itk_mask_recist = sitk.GetImageFromArray(segs_3D_recist)
        
        # Set metadata
        itk_img.SetSpacing(spacing)
        itk_img.SetDirection(direction.flatten())
        itk_img.SetOrigin(origin)
        
        # Save files with strategy suffix
        img_path = os.path.join(pred_save_dir, f"{base}_img.nii.gz")
        mask_path_recist = os.path.join(pred_save_dir, f"{base}_mask_recist.nii.gz")
        
        sitk.WriteImage(itk_img, img_path)
        sitk.WriteImage(itk_mask_recist, mask_path_recist)
        
       
        # Calculate metrics for RECIST
        dsc_recist = float('nan')
        nsd_recist = float('nan')
        if "gts" in data and data["gts"].max() > 0:
            metrics = calculate_metrics(segs_3D_recist, gts3d, spacing)
            dsc_recist = metrics['DSC']
            nsd_recist = metrics['NSD']
            print(f"RECIST Metrics: DSC = {dsc_recist:.4f}, NSD = {nsd_recist:.4f}")
        
        # Store RECIST results
        seg_info_recist["npz_name"].append(f"{base}_mask_recist.nii.gz")
        seg_info_recist["key_slice_index"].append(int(zmid))
        seg_info_recist["DSC"].append(dsc_recist)
        seg_info_recist["NSD"].append(nsd_recist)
        seg_info_recist["strategy"].append(strategy_name)
        seg_info_recist["num_predictions"].append(len(point_prompt) // 2)  # Number of points
        
        recist_time = time.time() - start_time
        print(f"RECIST completed in: {recist_time:.2f} seconds")

         # Create RECIST visualization
        viz_path = os.path.join(pred_save_dir, f"{base}_recist_visualization.png")
        visualize_and_handle_output(img3d, gts3d, segs_3D_recist, spacing, origin, direction, viz_path,
            dsc=dsc_recist,nsd=nsd_recist,time_taken=recist_time)
        
    

    
    # Record total case completion time
    case_end_time = time.time()
    case_total_time = case_end_time - total_start_time
    case_times.append(case_total_time)
    print(f"Total case completed in: {case_total_time:.2f} seconds")
    
    # Compare results if both strategies ran
    if run_medyolo and run_recist and 'dsc_medyolo' in locals() and 'dsc_recist' in locals():
        if not np.isnan(dsc_medyolo) and not np.isnan(dsc_recist):
            print(f"\n--- COMPARISON ---")
            print(f"MedYOLO: DSC={dsc_medyolo:.4f}, NSD={nsd_medyolo:.4f}")
            print(f"RECIST:  DSC={dsc_recist:.4f}, NSD={nsd_recist:.4f}")
            if dsc_medyolo > dsc_recist:
                print(f"MedYOLO performed better by {dsc_medyolo - dsc_recist:.4f} DSC")
            elif dsc_recist > dsc_medyolo:
                print(f"RECIST performed better by {dsc_recist - dsc_medyolo:.4f} DSC")
            else:
                print("Both strategies performed equally")
    
    print(f"\n{'='*60}\n")



# Save metrics summary
print("\n" + "="*50)
print("EVALUATION METRICS SUMMARY")
print("="*50)

# Save results for both strategies
if run_medyolo and len(seg_info_medyolo["npz_name"]) > 0:
    print("\n--- MEDYOLO RESULTS ---")
    metrics_df_medyolo = pd.DataFrame(seg_info_medyolo)
    valid_dsc_medyolo = [x for x in seg_info_medyolo["DSC"] if not np.isnan(x)]
    valid_nsd_medyolo = [x for x in seg_info_medyolo["NSD"] if not np.isnan(x)]
    
    if valid_dsc_medyolo:
        print(f"MedYOLO Average DSC: {np.mean(valid_dsc_medyolo):.4f}")
        print(f"MedYOLO Average NSD: {np.mean(valid_nsd_medyolo):.4f}")
    
    csv_path_medyolo = os.path.join(pred_save_dir, "evaluation_metrics_medyolo.csv")
    metrics_df_medyolo.to_csv(csv_path_medyolo, index=False)
    print(f"MedYOLO detailed metrics saved to: {csv_path_medyolo}")

if run_recist and len(seg_info_recist["npz_name"]) > 0:
    print("\n--- RECIST RESULTS ---")
    metrics_df_recist = pd.DataFrame(seg_info_recist)
    valid_dsc_recist = [x for x in seg_info_recist["DSC"] if not np.isnan(x)]
    valid_nsd_recist = [x for x in seg_info_recist["NSD"] if not np.isnan(x)]
    
    if valid_dsc_recist:
        print(f"RECIST Average DSC: {np.mean(valid_dsc_recist):.4f}")
        print(f"RECIST Average NSD: {np.mean(valid_nsd_recist):.4f}")
    
    csv_path_recist = os.path.join(pred_save_dir, "evaluation_metrics_recist.csv")
    metrics_df_recist.to_csv(csv_path_recist, index=False)
    print(f"RECIST detailed metrics saved to: {csv_path_recist}")

# Compare strategies if both ran
if run_medyolo and run_recist and len(valid_dsc_medyolo) > 0 and len(valid_dsc_recist) > 0:
    print("\n--- STRATEGY COMPARISON ---")
    medyolo_dsc = np.mean(valid_dsc_medyolo)
    recist_dsc = np.mean(valid_dsc_recist)
    medyolo_nsd = np.mean(valid_nsd_medyolo)
    recist_nsd = np.mean(valid_nsd_recist)
    
    print(f"MedYOLO: DSC={medyolo_dsc:.4f}, NSD={medyolo_nsd:.4f}")
    print(f"RECIST:  DSC={recist_dsc:.4f}, NSD={recist_nsd:.4f}")
    
    if medyolo_dsc > recist_dsc:
        print(f"MedYOLO performed better by {medyolo_dsc - recist_dsc:.4f} DSC")
    elif recist_dsc > medyolo_dsc:
        print(f"RECIST performed better by {recist_dsc - medyolo_dsc:.4f} DSC")
    else:
        print("Both strategies performed equally")


# Save detailed results to CSV
# Note: Individual strategy results are already saved above
print(f"\nDetailed metrics saved to individual CSV files")

# Timing summary
total_end_time = time.time()
total_time = total_end_time - total_start_time
valid_case_times = [t for t in case_times if t > 0]

print("\n" + "="*50)
print("TIMING SUMMARY")
print("="*50)
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Number of cases processed: {len(case_times)}")
print(f"Successful cases: {len(valid_case_times)}")
print(f"Average time per case: {np.mean(valid_case_times):.2f} ± {np.std(valid_case_times):.2f} seconds")
print(f"Fastest case: {np.min(valid_case_times):.2f} seconds")
print(f"Slowest case: {np.max(valid_case_times):.2f} seconds")
print(f"Total throughput: {len(valid_case_times)/total_time:.2f} cases/second")
print("="*50)
# Display top and bottom performers for MedYOLO
if run_medyolo and len(valid_dsc_medyolo) > 0:
    print(f"\nTop 3 MedYOLO DSC performers:")
    top_dsc_indices = np.argsort(valid_dsc_medyolo)[-3:][::-1]
    for i, idx in enumerate(top_dsc_indices):
        case_name = [name for name, dsc in zip(seg_info_medyolo["npz_name"], seg_info_medyolo["DSC"]) if not np.isnan(dsc)][idx]
        print(f"  {i+1}. {case_name}: DSC = {valid_dsc_medyolo[idx]:.4f}")

    print(f"\nTop 3 MedYOLO NSD performers (lowest is best):")
    top_nsd_indices = np.argsort(valid_nsd_medyolo)[:3]
    for i, idx in enumerate(top_nsd_indices):
        case_name = [name for name, nsd in zip(seg_info_medyolo["npz_name"], seg_info_medyolo["NSD"]) if not np.isnan(nsd)][idx]
        print(f"  {i+1}. {case_name}: NSD = {valid_nsd_medyolo[idx]:.4f}")

# Display top and bottom performers for RECIST
if run_recist and len(valid_dsc_recist) > 0:
    print(f"\nTop 3 RECIST DSC performers:")
    top_dsc_indices = np.argsort(valid_dsc_recist)[-3:][::-1]
    for i, idx in enumerate(top_dsc_indices):
        case_name = [name for name, dsc in zip(seg_info_recist["npz_name"], seg_info_recist["DSC"]) if not np.isnan(dsc)][idx]
        print(f"  {i+1}. {case_name}: DSC = {valid_dsc_recist[idx]:.4f}")

    print(f"\nTop 3 RECIST NSD performers (lowest is best):")
    top_nsd_indices = np.argsort(valid_nsd_recist)[:3]
    for i, idx in enumerate(top_nsd_indices):
        case_name = [name for name, nsd in zip(seg_info_recist["npz_name"], seg_info_recist["NSD"]) if not np.isnan(nsd)][idx]
        print(f"  {i+1}. {case_name}: NSD = {valid_nsd_recist[idx]:.4f}")

print("="*50)

