import csv
import os
import subprocess
import sys
from tqdm import tqdm # Import tqdm

# Define input file and output directory
input_csv_path = 'results/og_runs/NCT_predictions.csv'
output_dir = 'nct_csv_data'
api_base_url = "https://clinicaltrials.gov/api/v2/studies/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Processing {input_csv_path}...")
print(f"Saving downloaded CSVs to {output_dir}/")

# --- Pre-scan to collect all NCT IDs ---
all_nct_ids = []
print("Pre-scanning CSV to collect NCT IDs...")
try:
    with open(input_csv_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        if 'correct_nct' not in reader.fieldnames:
             print(f"Error: Column 'correct_nct' not found in {input_csv_path}", file=sys.stderr)
             sys.exit(1)
        for row in reader:
            nct_id_string = row.get('correct_nct', '').strip()
            if nct_id_string:
                ids_in_row = [nid.strip() for nid in nct_id_string.split(',') if nid.strip()]
                all_nct_ids.extend(ids_in_row)
    # Remove duplicates if necessary, though downloading duplicates might be okay
    # all_nct_ids = list(set(all_nct_ids)) 
    print(f"Found {len(all_nct_ids)} NCT IDs to download.")

except FileNotFoundError:
    print(f"Error: Input file not found at {input_csv_path}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during pre-scan: {e}", file=sys.stderr)
    sys.exit(1)

# --- Main download loop with tqdm ---
processed_count = 0
error_count = 0
print("Starting download process...")
if not all_nct_ids:
    print("No NCT IDs found to download.")
else:
    # Wrap the collected list with tqdm
    for nct_id in tqdm(all_nct_ids, desc="Downloading NCT data", unit="file"):
        # Construct the API URL and output file path
        api_url = f"{api_base_url}{nct_id}?format=csv"
        output_file_path = os.path.join(output_dir, f"{nct_id}.csv")

        # Skip if file already exists (optional, uncomment to enable)
        # if os.path.exists(output_file_path):
        #     # print(f"Skipping {nct_id}, file already exists.")
        #     processed_count += 1 # Count it as processed if skipping
        #     continue

        # Construct the curl command
        curl_command = [ # Corrected indentation
            'curl',
            '-X', 'GET',
            api_url,
            '-H', 'accept: text/csv',
            '-o', output_file_path, # Save output directly to file
            '-s', # Silent mode (don't show progress)
            '-L'  # Follow redirects
        ]

        # print(f"Downloading data for {nct_id} (from row {row_num}) to {output_file_path}...") # Original print removed
        try:
            result = subprocess.run(curl_command, check=True, capture_output=True, text=True)
            # Check if the downloaded file is empty or contains an error message (optional)
            if os.path.exists(output_file_path) and os.path.getsize(output_file_path) == 0:
                 # Use tqdm.write for messages inside the loop to avoid messing up the bar
                 tqdm.write(f"Warning: Downloaded file for {nct_id} is empty.")
                 # Optionally remove the empty file: os.remove(output_file_path)
            processed_count += 1
        except subprocess.CalledProcessError as e:
            tqdm.write(f"Error downloading data for {nct_id}: {e}") # Use tqdm.write
            # tqdm.write(f"Command output: {e.stdout}") # Often empty with -s
            # tqdm.write(f"Command error: {e.stderr}") # Often empty with -s
            error_count += 1 # Corrected indentation
            # Clean up potentially incomplete file
            if os.path.exists(output_file_path): # Corrected indentation
                os.remove(output_file_path)
        except Exception as e: # Corrected indentation
             tqdm.write(f"An unexpected error occurred for {nct_id}: {e}") # Use tqdm.write
             error_count += 1
             if os.path.exists(output_file_path):
                os.remove(output_file_path) # Corrected indentation

# --- Final Summary ---
# This print statement remains outside the loops
print(f"\nProcessing complete.")
print(f"Successfully processed/downloaded: {processed_count}") # Updated label slightly
print(f"Errors encountered: {error_count}")

# Removed original exception handling as it's now split between pre-scan and download
# except FileNotFoundError:
#     print(f"Error: Input file not found at {input_csv_path}", file=sys.stderr)
#     sys.exit(1)
# except Exception as e:
#     print(f"An unexpected error occurred: {e}", file=sys.stderr)
#     sys.exit(1)
# Removed stray continue
