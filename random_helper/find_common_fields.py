import csv
import os
import sys

# Directory containing the downloaded CSV files
data_dir = 'nct_csv_data'

if not os.path.isdir(data_dir):
    print(f"Error: Directory '{data_dir}' not found. Please run the download script first.", file=sys.stderr)
    sys.exit(1)

csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and os.path.isfile(os.path.join(data_dir, f))]

if not csv_files:
    print(f"Error: No CSV files found in '{data_dir}'.", file=sys.stderr)
    sys.exit(1)

from collections import Counter # Import Counter

print(f"Found {len(csv_files)} CSV files in '{data_dir}'. Analyzing headers...")

header_counts = Counter() # Use Counter to store header frequencies
processed_files = 0
files_with_errors = []

for filename in csv_files:
    filepath = os.path.join(data_dir, filename)
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as infile:
            # Handle potential empty files or files with only headers
            first_line = infile.readline()
            if not first_line:
                print(f"Warning: Skipping empty file '{filename}'.")
                continue
            
            # Reset file pointer and read header
            infile.seek(0)
            reader = csv.reader(infile)
            header = next(reader, None) # Read the first row as header

            if header is None:
                 print(f"Warning: Skipping file '{filename}' as it seems to have no header.")
                 continue
            
            # Increment count for each header found in this file
            for field in header:
                # Normalize field name (optional, e.g., lowercasing)
                # normalized_field = field.strip().lower() 
                # header_counts[normalized_field] += 1
                header_counts[field.strip()] += 1 # Count stripped field names

            processed_files += 1

    except Exception as e:
        print(f"Error processing file '{filename}': {e}", file=sys.stderr)
        files_with_errors.append(filename)
        # Continue processing other files even if one fails

# --- Analysis of header counts ---
print(f"\nAnalyzed {processed_files} files.")
if files_with_errors:
    print(f"Encountered errors in {len(files_with_errors)} files.")

if processed_files > 0:
    threshold_percentage = 95.0
    min_occurrences = int((threshold_percentage / 100.0) * processed_files)
    
    print(f"Finding fields present in > {threshold_percentage}% of analyzed files (minimum {min_occurrences} occurrences)...")

    frequent_fields = [
        field for field, count in header_counts.items() if count >= min_occurrences
    ]

    if frequent_fields:
        print(f"\nFields found in >{threshold_percentage}% of the {processed_files} analyzed files:")
        # Sort for consistent output
        sorted_fields = sorted(frequent_fields)
        for field in sorted_fields:
            count = header_counts[field]
            percentage = (count / processed_files) * 100
            print(f"- {field} (found in {count}/{processed_files} files, {percentage:.2f}%)")
    else:
        print(f"No fields were found in more than {threshold_percentage}% of the analyzed files.")
else:
     print("No CSV files could be successfully processed.")
