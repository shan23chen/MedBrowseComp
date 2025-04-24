#!/usr/bin/env python3
"""
CSV Splitter for Docker Container Prompts

This script splits a master CSV file containing prompts into multiple CSV files,
one for each container, to enable parallel execution of prompts.

Usage:
    python split_csv.py --input prompts_master.csv --output ./shared_docker_volume --containers 4 --prefix "computer-use-demo-instance-"
"""

import argparse
import csv
import os
import math
from pathlib import Path


def split_csv(input_file, output_dir, num_containers, container_prefix):
    """
    Split a CSV file into multiple CSV files for each container.
    
    Args:
        input_file (str): Path to the input CSV file
        output_dir (str): Directory where output CSV files will be saved
        num_containers (int): Number of containers to split for
        container_prefix (str): Prefix for container names
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read all prompts from the input CSV
    all_prompts = []
    header = None
    
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Check if there's a header row
        try:
            first_row = next(reader)
            # If the first row contains "prompt" in any of its cells, treat it as a header
            if any("prompt" in cell.lower() for cell in first_row):
                header = first_row
            else:
                # If not a header, add it to prompts
                all_prompts.append(first_row)
        except StopIteration:
            # Empty file
            print(f"Warning: Input file {input_file} is empty")
            return
        
        # Read the rest of the prompts
        for row in reader:
            if row:  # Skip empty rows
                all_prompts.append(row)
    
    # If no prompts were found, exit
    if not all_prompts:
        print(f"No prompts found in {input_file}")
        return
    
    # Calculate prompts per container (rounded up to ensure all prompts are assigned)
    prompts_per_container = math.ceil(len(all_prompts) / num_containers)
    
    print(f"Found {len(all_prompts)} prompts. Splitting into {num_containers} containers.")
    print(f"Each container will get approximately {prompts_per_container} prompts.")
    
    # Split prompts among containers
    for container_idx in range(num_containers):
        container_name = f"{container_prefix}{container_idx + 1}"
        output_file = output_path / f"{container_name}.csv"
        
        # Calculate start and end indices for this container's prompts
        start_idx = container_idx * prompts_per_container
        end_idx = min(start_idx + prompts_per_container, len(all_prompts))
        
        # Get prompts for this container
        container_prompts = all_prompts[start_idx:end_idx]
        
        # Skip if no prompts for this container
        if not container_prompts:
            print(f"No prompts allocated for container {container_name}, skipping")
            continue
        
        # Write prompts to CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header if exists
            if header:
                writer.writerow(header)
            
            # Write prompts
            writer.writerows(container_prompts)
        
        print(f"Created {output_file} with {len(container_prompts)} prompts for container {container_name}")


def main():
    """Parse arguments and run the CSV splitter"""
    parser = argparse.ArgumentParser(description='Split a CSV file into multiple files for container prompts')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file with prompts')
    parser.add_argument('--output', '-o', required=True, help='Output directory for split CSV files')
    parser.add_argument('--containers', '-c', type=int, required=True, help='Number of containers to split for')
    parser.add_argument('--prefix', '-p', default='computer-use-demo-instance-', 
                        help='Container name prefix (default: computer-use-demo-instance-)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1
    
    if args.containers <= 0:
        print("Error: Number of containers must be positive")
        return 1
    
    # Split the CSV
    split_csv(args.input, args.output, args.containers, args.prefix)
    return 0


if __name__ == "__main__":
    exit(main())