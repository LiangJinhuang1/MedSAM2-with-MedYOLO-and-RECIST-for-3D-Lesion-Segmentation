#!/usr/bin/env python3
"""
Script to parse training logs and plot loss trends
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

def plot_training_results(json_file_path="FLARE_results/logs/train_stats.json"):
    """
    Plot training results from JSON file
    """
    print(f"Parsing training stats from: {json_file_path}")
    
    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} not found!")
        return
    
    # Initialize data containers
    epochs = []
    steps = []
    losses = []
    loss_components = {
        'total_loss': [],
        'loss_mask': [],
        'loss_dice': [],
        'loss_iou': [],
        'loss_class': [],
        'core_loss': []
    }
    
    try:
        with open(json_file_path, 'r') as f:
            # Handle JSONL format (one JSON object per line)
            data = []
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data.append(json.loads(line))
        
        # Extract data from JSON
        for entry in data:
            # Extract step number
            if 'Trainer/steps_train' in entry:
                current_step = int(entry['Trainer/steps_train'])
                steps.append(current_step)
            
            # Extract epoch number
            if 'Trainer/epoch' in entry:
                current_epoch = int(entry['Trainer/epoch'])
                epochs.append(current_epoch)
            
            # Extract loss values
            loss_mappings = {
                'total_loss': 'Losses/train_all_loss',
                'loss_mask': 'Losses/train_all_loss_mask',
                'loss_dice': 'Losses/train_all_loss_dice',
                'loss_iou': 'Losses/train_all_loss_iou',
                'loss_class': 'Losses/train_all_loss_class',
                'core_loss': 'Losses/train_all_core_loss'
            }
            
            for loss_type, json_key in loss_mappings.items():
                if json_key in entry:
                    loss_value = float(entry[json_key])
                    loss_components[loss_type].append(loss_value)
            
            # Use total loss as the main loss
            if 'Losses/train_all_loss' in entry:
                losses.append(float(entry['Losses/train_all_loss']))
        
        print(f"Found {len(losses)} loss values from training stats")
        
        if not losses:
            print("No loss data found. Check your JSON file format.")
            return
        
        # Get the final epoch number from the last line only
        final_epoch = data[-1]['Trainer/epoch'] if data else 0
        print(f"Final epoch from last line: {final_epoch}")
        
        # Plot the last N+1 epochs where N is the final epoch number
        epochs_to_plot = final_epoch + 1
        if len(losses) > epochs_to_plot:
            start_idx = len(losses) - epochs_to_plot
            losses = losses[start_idx:]
            steps = steps[start_idx:]
            epochs = epochs[start_idx:]
            for loss_type in loss_components:
                if loss_components[loss_type]:
                    loss_components[loss_type] = loss_components[loss_type][start_idx:]
            print(f"Plotting last {epochs_to_plot} epochs (epochs {epochs[0]} to {epochs[-1]})")
        else:
            print(f"Plotting all {len(losses)} epochs")
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'MedSAM2 Training Results (Last {len(losses)} epochs)', fontsize=16, fontweight='bold')
        
        # Plot 1: Main loss over steps
        axes[0, 0].plot(steps, losses, 'b-', linewidth=1, alpha=0.8)
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Training Loss Over Steps')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Main loss over epochs
        if epochs:
            axes[0, 1].plot(epochs, losses, 'r-', linewidth=1, alpha=0.8)
            axes[0, 1].set_xlabel('Epochs')
            axes[0, 1].set_ylabel('Total Loss')
            axes[0, 1].set_title('Training Loss Over Epochs')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: All loss components over steps
        for loss_type, values in loss_components.items():
            if values and len(values) == len(steps):
                axes[1, 0].plot(steps, values, label=loss_type, linewidth=1, alpha=0.8)
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Loss Values')
        axes[1, 0].set_title('All Loss Components Over Steps')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Loss distribution
        axes[1, 1].hist(losses, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Loss Values')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Loss Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = "training_results_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training results plot saved to: {output_path}")
        
        # Display summary statistics
        print("\nTraining Summary:")
        print(f"Total training steps: {len(steps)}")
        print(f"Total epochs: {len(set(epochs)) if epochs else 'N/A'}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Best loss: {min(losses):.4f}")
        print(f"Average loss: {np.mean(losses):.4f}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error parsing JSON file: {e}")

if __name__ == "__main__":
    plot_training_results() 