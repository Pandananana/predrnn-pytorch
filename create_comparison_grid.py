#!/usr/bin/env python3
"""
Script to create a comparison grid of ground truth and predicted images.
Dynamically detects the number of GT images and creates a 2-row grid:
- Top row: All ground truth images
- Bottom row: Ground truth where no prediction exists, predictions where available
"""

import os
from PIL import Image
import argparse


def create_comparison_grid(input_dir, output_path):
    """
    Create a 2-row grid comparing ground truth and predictions.
    Dynamically determines the number of columns based on available GT images.
    
    Args:
        input_dir: Directory containing GT*.png and PD*.png files
        output_path: Path where the output image will be saved
    """
    # Load images
    print(f"Loading images from {input_dir}...")
    
    # Find all GT images to determine the number of columns
    gt_files = {}
    pd_files = {}
    
    for filename in os.listdir(input_dir):
        if filename.upper().startswith('GT') and filename.lower().endswith('.png'):
            # Extract the number from the filename (e.g., GT1.png -> 1)
            try:
                num_str = filename[2:-4]  # Remove 'GT' prefix and '.png' suffix
                num = int(num_str)
                gt_files[num] = os.path.join(input_dir, filename)
            except ValueError:
                continue
        elif filename.upper().startswith('PD') and filename.lower().endswith('.png'):
            # Extract the number from the filename (e.g., PD6.png -> 6)
            try:
                num_str = filename[2:-4]  # Remove 'PD' prefix and '.png' suffix
                num = int(num_str)
                pd_files[num] = os.path.join(input_dir, filename)
            except ValueError:
                continue
    
    if not gt_files:
        raise FileNotFoundError(f"No ground truth images found in {input_dir}")
    
    # Determine the range of indices
    indices = sorted(gt_files.keys())
    num_cols = len(indices)
    
    print(f"Found {num_cols} ground truth images: {indices}")
    print(f"Found {len(pd_files)} prediction images: {sorted(pd_files.keys())}")
    
    # Top row: All GT images
    top_row = []
    for idx in indices:
        img = Image.open(gt_files[idx])
        top_row.append(img)
    
    # Bottom row: GT for indices without predictions, PD where predictions exist
    bottom_row = []
    bottom_row_labels = []
    for idx in indices:
        if idx in pd_files:
            img = Image.open(pd_files[idx])
            bottom_row.append(img)
            bottom_row_labels.append(f"PD{idx}")
        else:
            img = Image.open(gt_files[idx])
            bottom_row.append(img)
            bottom_row_labels.append(f"GT{idx}")
    
    print(f"Bottom row composition: {', '.join(bottom_row_labels)}")
    
    # Get dimensions (assuming all images are the same size)
    img_width, img_height = top_row[0].size
    print(f"Individual image size: {img_width}x{img_height}")
    
    # Add spacing between images and rows for better visualization
    spacing = 5
    grid_width_with_spacing = img_width * num_cols + spacing * (num_cols - 1)
    grid_height_with_spacing = img_height * 2 + spacing
    
    # Create white canvas
    grid_image = Image.new('RGB', (grid_width_with_spacing, grid_height_with_spacing), 'white')
    
    # Paste top row (all ground truth)
    print(f"Creating top row (Ground Truth for all {num_cols} frames)...")
    for i, img in enumerate(top_row):
        x_offset = i * (img_width + spacing)
        grid_image.paste(img, (x_offset, 0))
    
    # Paste bottom row (GT where no prediction, PD where prediction exists)
    print("Creating bottom row (GT for input frames, PD for predicted frames)...")
    for i, img in enumerate(bottom_row):
        x_offset = i * (img_width + spacing)
        y_offset = img_height + spacing
        grid_image.paste(img, (x_offset, y_offset))
    
    # Save the result
    print(f"Saving comparison grid to {output_path}...")
    grid_image.save(output_path, quality=95)
    print(f"Done! Comparison grid saved to: {output_path}")
    print(f"Grid dimensions: {grid_width_with_spacing}x{grid_height_with_spacing}")


def main():
    parser = argparse.ArgumentParser(
        description='Create a comparison grid of ground truth and predicted images'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='results/temperature_predrnn_v2/1000/1',
        help='Directory containing the images (default: results/temperature_predrnn_v2/1000/1)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path if relative
    input_dir = os.path.abspath(args.input_dir)
    output_path = os.path.abspath(args.input_dir + '/comparison.png')
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    # Create comparison grid
    try:
        create_comparison_grid(input_dir, output_path)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

