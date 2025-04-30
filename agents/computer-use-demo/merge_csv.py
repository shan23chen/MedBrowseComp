#!/usr/bin/env python3
"""
CSV Merger for Container Prompt Results

This script merges multiple partial CSV files from containers back into 
a complete CSV file with all prompts and their results.

Usage:
    python merge_csv.py --input ./shared_docker_volume --output ./results.csv --prefix "computer-use-demo-instance-"
"""

import argparse
import csv
import os
import time
import glob
from pathlib import Path


def wait_for_containers(input_dir, file_pattern, timeout=3600, check_interval=10):
    """
    Wait for all containers to finish processing their CSV files.
    
    Args:
        input_dir (str): Directory containing the CSV files
        file_pattern (str): Pattern to match CSV files
        timeout (int): Maximum time to wait in seconds
        check_interval (int): Time between checks in seconds
    
    Returns:
        list: List of completed CSV files
    """
    input_path = Path(input_dir)
    
    # Find all CSV files matching the pattern
    csv_files = list(input_path.glob(file_pattern))
    if not csv_files:
        print(f"No CSV files found matching pattern '{file_pattern}' in {input_dir}")
        return []
    
    print(f"Found {len(csv_files)} CSV files. Waiting for processing to complete...")
    
    # Initialize tracking variables
    start_time = time.time()
    completed_files = []
    pending_files = {str(csv_file) for csv_file in csv_files}
    
    # Wait for all files to be completed
    while pending_files and (time.time() - start_time) < timeout:
        # Check for completion marker files
        for csv_file in list(pending_files):
            completed_marker = f"{csv_file}.completed"
            
            # Method 1: Check for .completed marker file
            if os.path.exists(completed_marker):
                print(f"Container completed: {os.path.basename(csv_file)}")
                completed_files.append(csv_file)
                pending_files.remove(csv_file)
                continue
                
            # Method 2: Check if the CSV has results column populated for all rows
            try:
                with open(csv_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    
                    # Check if file has a header
                    has_header = False
                    if rows and "prompt" in rows[0][0].lower():
                        has_header = True
                    
                    # Skip header if it exists
                    data_rows = rows[1:] if has_header else rows
                    
                    # Check if all rows have at least 2 columns (prompt and result)
                    all_have_results = all(len(row) >= 2 for row in data_rows)
                    
                    if all_have_results and data_rows:
                        print(f"Container completed (all rows have results): {os.path.basename(csv_file)}")
                        completed_files.append(csv_file)
                        pending_files.remove(csv_file)
                        
                        # Create a completion marker for future reference
                        with open(completed_marker, 'w') as marker:
                            marker.write(f"Completed processing at {time.ctime()}")
            except Exception as e:
                print(f"Error checking CSV file {csv_file}: {e}")
        
        # If there are still pending files, wait and check again
        if pending_files:
            remaining = len(pending_files)
            elapsed = int(time.time() - start_time)
            print(f"Waiting for {remaining} containers to complete ({elapsed}s elapsed)...")
            time.sleep(check_interval)
    
    # Check if we timed out
    if pending_files:
        print(f"WARNING: Timed out waiting for {len(pending_files)} containers to complete!")
        print("Proceeding with partial results.")
        
    # Return the list of completed files (and add any that are still pending)
    return completed_files + list(pending_files)


def merge_csv_files(csv_files, output_file):
    """
    Merge multiple CSV files into a single output file.
    
    Args:
        csv_files (list): List of CSV file paths
        output_file (str): Path to the output merged CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not csv_files:
        print("No CSV files to merge.")
        return False
    
    try:
        # First, collect all rows from all files
        all_rows = []
        has_header = False
        header = None
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    
                    # Check if the file has a header
                    file_has_header = False
                    if rows and len(rows[0]) > 0 and "prompt" in rows[0][0].lower():
                        file_has_header = True
                        has_header = True
                        
                        # Use the first header we encounter as our header
                        if header is None:
                            header = rows[0]
                            if len(header) < 2:  # Ensure header has result column
                                header.append("result")
                    
                    # Add data rows (skipping header if present)
                    data_rows = rows[1:] if file_has_header else rows
                    all_rows.extend([(csv_file, row) for row in data_rows])
                
                print(f"Processed {os.path.basename(csv_file)}: {len(data_rows)} rows")
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        # Sort rows by the numeric portion of the filename (e.g., instance-1.csv comes before instance-2.csv)
        # This assumes that CSVs are named in the format prefix-number.csv
        def extract_number(filename):
            try:
                # Extract the numeric part from the filename
                import re
                match = re.search(r'(\d+)', os.path.basename(filename))
                if match:
                    return int(match.group(1))
                else:
                    return float('inf')  # Put files without numbers at the end
            except:
                return float('inf')
        
        # Sort by filename first to keep related rows together
        all_rows.sort(key=lambda x: extract_number(x[0]))
        
        # Write the merged CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if needed
            if has_header and header:
                writer.writerow(header)
            
            # Write all data rows
            for _, row in all_rows:
                # Ensure all rows have at least two columns (prompt and result)
                if len(row) < 1:
                    row.append("")  # Add empty prompt if missing
                if len(row) < 2:
                    row.append("")  # Add empty result if missing
                writer.writerow(row)
        
        print(f"Successfully merged {len(csv_files)} CSV files into {output_file}")
        print(f"Total rows in merged file: {len(all_rows)}")
        return True
        
    except Exception as e:
        print(f"Error merging CSV files: {e}")
        return False


def main():
    """Parse arguments and run the CSV merger"""
    parser = argparse.ArgumentParser(description='Merge partial CSV files into a complete results file')
    parser.add_argument('--input', '-i', required=True, help='Input directory with partial CSV files')
    parser.add_argument('--output', '-o', required=True, help='Output path for merged CSV file')
    parser.add_argument('--prefix', '-p', default='computer-use-demo-instance-', 
                        help='Container/CSV name prefix (default: computer-use-demo-instance-)')
    parser.add_argument('--wait', '-w', action='store_true', 
                        help='Wait for all containers to complete processing')
    parser.add_argument('--timeout', '-t', type=int, default=3600,
                        help='Maximum time to wait for containers (in seconds, default: 3600)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isdir(args.input):
        print(f"Error: Input directory {args.input} not found")
        return 1
    
    # Create CSV file pattern
    csv_pattern = f"{args.prefix}*.csv"
    
    # Wait for containers to complete if requested
    if args.wait:
        csv_files = wait_for_containers(args.input, csv_pattern, args.timeout)
    else:
        # Otherwise, just find all matching CSV files
        csv_files = glob.glob(os.path.join(args.input, csv_pattern))
        
    # Make sure we found some CSV files
    if not csv_files:
        print(f"No CSV files found matching pattern '{csv_pattern}' in {args.input}")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Merge the CSV files
    success = merge_csv_files(csv_files, args.output)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())