import pandas as pd
import re
import argparse

def extract_nct_numbers(df, n=None):
    """
    Extracts NCT numbers from the evidence column, adds them to a new column,
    and removes them from the original text. Only keeps rows with NCT numbers.
    
    Args:
        df (pandas.DataFrame): The input dataframe
        n (int, optional): Number of rows to process. If None, process all rows.
    
    Returns:
        pandas.DataFrame: The modified dataframe with only NCT-containing rows
    """
    # Create a copy of the dataframe to work with
    result_df = df.copy()
    
    # Limit to first n rows if specified
    if n is not None:
        result_df = result_df.iloc[:n]
    
    # Add a new NCT column
    result_df['NCT'] = ''
    
    # Define regex pattern for NCT numbers (NCT followed by 8 digits)
    nct_pattern = r'NCT\d{8}'
    
    # List to store indices of rows with NCT numbers
    valid_indices = []
    
    # Process each row
    for idx, row in result_df.iterrows():
        evidence = row['evidence']
        
        # Find all NCT numbers in the evidence text
        nct_numbers = re.findall(nct_pattern, evidence)
        
        if nct_numbers:
            # Remove duplicates while preserving order
            unique_nct = []
            for nct in nct_numbers:
                if nct not in unique_nct:
                    unique_nct.append(nct)
            
            # Join unique NCT numbers with comma
            result_df.at[idx, 'NCT'] = ', '.join(unique_nct)
            
            # Remove NCT numbers from evidence text
            clean_evidence = evidence
            for nct in unique_nct:
                # Replace the exact NCT number
                clean_evidence = re.sub(r'\(?' + re.escape(nct) + r'\)?', '', clean_evidence)
                # Clean up any artifacts like "ClinicalTrials.gov Identifier: " or double spaces
                clean_evidence = re.sub(r'ClinicalTrials\.gov Identifier:\s*[,.]?\s*', '', clean_evidence)
                clean_evidence = re.sub(r'\s+', ' ', clean_evidence)
                # Clean up any trailing periods if we removed the last part
                clean_evidence = re.sub(r'\s+\.', '.', clean_evidence)
            
            result_df.at[idx, 'evidence'] = clean_evidence.strip()
            valid_indices.append(idx)
    
    # Keep only rows with NCT numbers
    result_df = result_df.loc[valid_indices]
    
    # Remove duplicate evidence columns
    result_df = result_df.drop_duplicates(subset=['evidence'])
    
    return result_df

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Extract NCT numbers from evidence column in a CSV file')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output CSV file path')
    parser.add_argument('-n', type=int, help='Process only the first n rows (for testing)')
    args = parser.parse_args()
    
    try:
        # Read the CSV file
        df = pd.read_csv(args.input_file)
        
        # Process the data
        result_df = extract_nct_numbers(df, args.n)
        
        # Save the result
        result_df.to_csv(args.output_file, index=False)
        
        print(f"Found {len(result_df)} rows with NCT numbers (after removing duplicates).")
        print(f"Output saved to {args.output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()