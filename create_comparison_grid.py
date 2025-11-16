#!/usr/bin/env python3
"""
Script to create a comparison grid of ground truth and predicted images.
Top row: Ground truth 1-8
Bottom row: Ground truth 1-5, Predicted 6-8
"""

import os
from PIL import Image
import argparse


def create_comparison_grid(input_dir, output_path):
    """
    Create a 2x8 grid comparing ground truth and predictions.
    
    Args:
        input_dir: Directory containing gt*.png and pd*.png files
        output_path: Path where the output image will be saved
    """
    # Load images
    print(f"Loading images from {input_dir}...")
    
    # Top row: gt1-gt8
    top_row = []
    for i in range(1, 9):
        img_path = os.path.join(input_dir, f"gt{i}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing ground truth image: {img_path}")
        top_row.append(Image.open(img_path))
    
    # Bottom row: gt1-gt5, pd6-pd8
    bottom_row = []
    for i in range(1, 6):
        img_path = os.path.join(input_dir, f"gt{i}.png")
        bottom_row.append(Image.open(img_path))
    
    for i in range(6, 9):
        img_path = os.path.join(input_dir, f"pd{i}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing prediction image: {img_path}")
        bottom_row.append(Image.open(img_path))
    
    # Get dimensions (assuming all images are the same size)
    img_width, img_height = top_row[0].size
    print(f"Individual image size: {img_width}x{img_height}")
    
    # Create the grid (2 rows x 8 columns)
    grid_width = img_width * 8
    grid_height = img_height * 2
    
    # Add spacing between images and rows for better visualization
    spacing = 5
    grid_width_with_spacing = img_width * 8 + spacing * 7
    grid_height_with_spacing = img_height * 2 + spacing
    
    # Create white canvas
    grid_image = Image.new('RGB', (grid_width_with_spacing, grid_height_with_spacing), 'white')
    
    # Paste top row (ground truth 1-8)
    print("Creating top row (Ground Truth 1-8)...")
    for i, img in enumerate(top_row):
        x_offset = i * (img_width + spacing)
        grid_image.paste(img, (x_offset, 0))
    
    # Paste bottom row (gt 1-5, pd 6-8)
    print("Creating bottom row (GT 1-5, Pred 6-8)...")
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

