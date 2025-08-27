"""
Utilities for Automatic Mask Generator (AMG) integration with SAM2 training.
Handles batch size mismatches and provides proper mask generation for unlabeled data.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from sam2.utils.amg import AutomaticMaskGenerator


def generate_masks_for_batch(
    amg_generator: AutomaticMaskGenerator,
    images: torch.Tensor,
    device: torch.device,
    max_masks_per_image: int = 10,
    min_mask_area: int = 100,
) -> List[torch.Tensor]:
    """
    Generate masks for a batch of images using AMG.
    
    Args:
        amg_generator: The AMG generator instance
        images: Batch of images with shape [B, C, H, W] or [B, H, W, C]
        device: Device to process on
        max_masks_per_image: Maximum number of masks to generate per image
        min_mask_area: Minimum area for masks to be considered valid
        
    Returns:
        List of mask tensors, each with shape [num_masks, H, W] for each image
    """
    batch_size = images.shape[0]
    all_masks = []
    
    for i in range(batch_size):
        # Extract single image
        if images.shape[1] == 3:  # [B, C, H, W] format
            image = images[i].permute(1, 2, 0)  # Convert to [H, W, C]
        else:  # [B, H, W, C] format
            image = images[i]
        
        # Convert to numpy and uint8 format
        if image.dtype == torch.float32:
            if image.max() <= 1.0:
                image_np = (image.cpu().numpy() * 255).astype(np.uint8)
            else:
                image_np = image.cpu().numpy().astype(np.uint8)
        else:
            image_np = image.cpu().numpy().astype(np.uint8)
        
        # Generate masks for this image
        try:
            mask_data = amg_generator.generate(image_np)
            
            # Extract masks from the result
            if "masks" in mask_data:
                masks = mask_data["masks"]
            elif "rles" in mask_data:
                # Convert RLE to binary masks
                masks = []
                for rle in mask_data["rles"]:
                    mask = rle_to_mask(rle)
                    masks.append(mask)
            else:
                masks = []
            
            # Filter masks by area and limit number
            valid_masks = []
            for mask in masks:
                if isinstance(mask, np.ndarray):
                    mask_tensor = torch.from_numpy(mask).bool()
                else:
                    mask_tensor = mask.bool()
                
                # Check area
                area = mask_tensor.sum().item()
                if area >= min_mask_area:
                    valid_masks.append(mask_tensor)
                
                if len(valid_masks) >= max_masks_per_image:
                    break
            
            # Stack masks for this image
            if valid_masks:
                image_masks = torch.stack(valid_masks, dim=0)  # [num_masks, H, W]
            else:
                # Create empty mask if no valid masks found
                H, W = image_np.shape[:2]
                image_masks = torch.zeros((1, H, W), dtype=torch.bool, device=device)
            
        except Exception as e:
            print(f"Error generating masks for image {i}: {e}")
            # Create empty mask on error
            H, W = image_np.shape[:2]
            image_masks = torch.zeros((1, H, W), dtype=torch.bool, device=device)
        
        all_masks.append(image_masks)
    
    return all_masks


def create_batched_masks(
    individual_masks: List[torch.Tensor],
    target_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a properly batched mask tensor from individual image masks.
    
    Args:
        individual_masks: List of mask tensors, each with shape [num_masks, H, W]
        target_batch_size: The desired batch size for the output
        device: Device to place the tensor on
        
    Returns:
        Batched mask tensor with shape [target_batch_size, 1, H, W]
    """
    if not individual_masks:
        raise ValueError("No masks provided")
    
    # Get the first mask to determine dimensions
    first_mask = individual_masks[0]
    H, W = first_mask.shape[-2:]
    
    # Create batched tensor
    batched_masks = torch.zeros((target_batch_size, 1, H, W), dtype=torch.bool, device=device)
    
    for i in range(min(len(individual_masks), target_batch_size)):
        image_masks = individual_masks[i]  # [num_masks, H, W]
        
        if image_masks.shape[0] > 0:
            # Use the first mask for this image (or combine multiple masks)
            if image_masks.shape[0] == 1:
                mask = image_masks[0]  # [H, W]
            else:
                # Combine multiple masks using OR operation
                mask = image_masks.any(dim=0)  # [H, W]
            
            # Ensure proper shape and add to batch
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # [1, H, W]
            
            batched_masks[i, 0] = mask
    
    return batched_masks


def expand_masks_to_batch_size(
    masks: torch.Tensor,
    target_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Expand or repeat masks to match the target batch size.
    
    Args:
        masks: Input mask tensor with shape [current_batch, 1, H, W]
        target_batch_size: The desired batch size
        device: Device to place the tensor on
        
    Returns:
        Expanded mask tensor with shape [target_batch_size, 1, H, W]
    """
    current_batch_size = masks.shape[0]
    
    if current_batch_size == target_batch_size:
        return masks
    
    if current_batch_size == 1:
        # Repeat the single mask to match target batch size
        return masks.repeat(target_batch_size, 1, 1, 1)
    
    elif current_batch_size < target_batch_size:
        # Repeat masks cyclically to reach target batch size
        repeats_needed = (target_batch_size + current_batch_size - 1) // current_batch_size
        expanded = masks.repeat(repeats_needed, 1, 1, 1)
        return expanded[:target_batch_size]
    
    else:
        # Truncate to target batch size
        return masks[:target_batch_size]


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Convert RLE to binary mask."""
    from sam2.utils.amg import rle_to_mask as _rle_to_mask
    return _rle_to_mask(rle)


def create_amg_generator(
    sam2_model,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    box_nms_thresh: float = 0.7,
    crop_n_layers: int = 0,
    crop_nms_thresh: float = 0.7,
    crop_overlap_ratio: float = 512 / 1500,
    crop_n_points_downscale_factor: int = 1,
    min_mask_region_area: int = 100,
    output_mode: str = "binary_mask",
) -> AutomaticMaskGenerator:
    """
    Create an AMG generator with default parameters suitable for training.
    
    Args:
        sam2_model: The SAM2 model to use for mask generation
        points_per_side: Number of points per side for point grid
        pred_iou_thresh: IoU threshold for filtering masks
        stability_score_thresh: Stability score threshold for filtering masks
        box_nms_thresh: NMS threshold for boxes
        crop_n_layers: Number of crop layers (0 for no cropping)
        crop_nms_thresh: NMS threshold for crops
        crop_overlap_ratio: Overlap ratio for crops
        crop_n_points_downscale_factor: Downscale factor for points in crops
        min_mask_region_area: Minimum area for mask regions
        output_mode: Output mode for masks
        
    Returns:
        Configured AMG generator
    """
    return AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        output_mode=output_mode,
    )


def integrate_amg_masks_into_training(
    images: torch.Tensor,
    existing_masks: Optional[torch.Tensor],
    amg_generator: AutomaticMaskGenerator,
    device: torch.device,
    use_amg_probability: float = 0.5,
    max_masks_per_image: int = 10,
    min_mask_area: int = 100,
) -> torch.Tensor:
    """
    Integrate AMG-generated masks into the training pipeline.
    
    Args:
        images: Input images with shape [B, C, H, W]
        existing_masks: Existing ground truth masks with shape [B, 1, H, W] or None
        amg_generator: AMG generator instance
        device: Device to process on
        use_amg_probability: Probability of using AMG masks instead of existing masks
        max_masks_per_image: Maximum masks per image for AMG
        min_mask_area: Minimum area for AMG masks
        
    Returns:
        Final mask tensor with shape [B, 1, H, W]
    """
    batch_size = images.shape[0]
    
    # Decide whether to use AMG for each image in the batch
    use_amg = torch.rand(batch_size, device=device) < use_amg_probability
    
    if not use_amg.any():
        # No AMG needed, return existing masks
        if existing_masks is not None:
            return expand_masks_to_batch_size(existing_masks, batch_size, device)
        else:
            # No masks available, create empty masks
            H, W = images.shape[-2:]
            return torch.zeros((batch_size, 1, H, W), dtype=torch.bool, device=device)
    
    # Generate AMG masks for images that need them
    amg_masks = generate_masks_for_batch(
        amg_generator, images, device, max_masks_per_image, min_mask_area
    )
    
    # Create final mask tensor
    H, W = images.shape[-2:]
    final_masks = torch.zeros((batch_size, 1, H, W), dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        if use_amg[i]:
            # Use AMG mask for this image
            if amg_masks[i].shape[0] > 0:
                # Combine multiple masks if present
                if amg_masks[i].shape[0] == 1:
                    final_masks[i, 0] = amg_masks[i][0]
                else:
                    final_masks[i, 0] = amg_masks[i].any(dim=0)
        else:
            # Use existing mask for this image
            if existing_masks is not None and i < existing_masks.shape[0]:
                final_masks[i, 0] = existing_masks[i, 0]
    
    return final_masks 