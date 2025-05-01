import pandas as pd
import argparse
import logging
import csv
import sys
import os
import traceback

# Increase CSV field size limit for potentially long prompts
max_int = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define prompt generation functions (copied and adapted from process_NCT_predictions.py logic provided)
def generate_ingredient_prompt(row):
    task_name = "Ingredient"
    # Ensure required columns exist and handle potential missing data gracefully
    nct = row.get('NCT', 'UNKNOWN_NCT')
    eff_group = row.get('effecacy_group', 'UNKNOWN_EFFICACY_GROUP')
    ingredient = row.get('Ingredient', 'UNKNOWN_INGREDIENT')

    if pd.isna(ingredient) or ingredient == 'UNKNOWN_INGREDIENT':
         logging.warning(f"Missing 'Ingredient' for NCT {nct}. Skipping Ingredient prompt generation.")
         return None # Skip if essential data is missing

    first_letter = ingredient[0] if isinstance(ingredient, str) and len(ingredient) > 0 else '?'

    if ingredient == 'APALUTAMIDE':
        prompt = f"For clinical trial {nct}. Among the {eff_group} effective regimen ingredients; find which is the ingredient with the first letter start with AP\n return only the name of the ingredient or unknown in all CAPS in the format of INGREDIENT: name"
    else:
        prompt = f"For clinical trial {nct}. Among the {eff_group} effective regimen ingredients; find which is the ingredient with the first letter start with {first_letter}\n return only the name of the ingredient or unknown in all CAPS in the format of INGREDIENT: name"
    gold = ingredient
    return {"prompt": prompt, "gold": gold, "task_name": task_name}

def generate_applicant_prompt(row):
    task_name = "Applicant_Full_Name"
    nct = row.get('NCT', 'UNKNOWN_NCT')
    eff_group = row.get('effecacy_group', 'UNKNOWN_EFFICACY_GROUP')
    ingredient = row.get('Ingredient', 'UNKNOWN_INGREDIENT')
    applicant = row.get('Applicant_Full_Name', 'UNKNOWN_APPLICANT')

    if pd.isna(ingredient) or ingredient == 'UNKNOWN_INGREDIENT':
         logging.warning(f"Missing 'Ingredient' for NCT {nct}. Skipping Applicant prompt generation.")
         return None
    if pd.isna(applicant): # Allow UNKNOWN_APPLICANT from get() default
         logging.warning(f"Missing 'Applicant_Full_Name' for NCT {nct}. Using '{applicant}'.")


    first_letter = ingredient[0] if isinstance(ingredient, str) and len(ingredient) > 0 else '?'

    if ingredient == 'APALUTAMIDE':
        prompt = f"For clinical trial {nct}. Among the {eff_group} effective regimen ingredients; find which is the ingredient with the first letter start with AP.\n\nThen, find which company has the latest FDA approval date up till Dec, 2024 for this identified ingredient. Note that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals. \nReturn only the company name in format of COMPANY: name."
    else:
        prompt = f"First, for clinical trial {nct} among the {eff_group} effective regimen ingredients, identify which ingredient starts with the letter {first_letter}.\n\nThen, find which company has the latest FDA approval date up till Dec, 2024 for this identified ingredient. Note that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals. \nReturn only the company name in format of COMPANY: name."
    gold = applicant
    return {"prompt": prompt, "gold": gold, "task_name": task_name}

def generate_patent_prompt(row):
    task_name = "Patent_Expire_Date_Text"
    nct = row.get('NCT', 'UNKNOWN_NCT')
    eff_group = row.get('effecacy_group', 'UNKNOWN_EFFICACY_GROUP')
    ingredient = row.get('Ingredient', 'UNKNOWN_INGREDIENT')
    patent_date = row.get('Patent_Expire_Date_Text', 'UNKNOWN_PATENT_DATE') # Use get for safety

    if pd.isna(ingredient) or ingredient == 'UNKNOWN_INGREDIENT':
         logging.warning(f"Missing 'Ingredient' for NCT {nct}. Skipping Patent prompt generation.")
         return None
    if pd.isna(patent_date): # Allow UNKNOWN_PATENT_DATE
         logging.warning(f"Missing 'Patent_Expire_Date_Text' for NCT {nct}. Using '{patent_date}'.")


    first_letter = ingredient[0] if isinstance(ingredient, str) and len(ingredient) > 0 else '?'

    if ingredient == 'APALUTAMIDE':
         prompt = f"For clinical trial {nct}. Among the {eff_group} effective regimen ingredients; find which is the ingredient with the first letter start with AP.\n\nThen, for this identified ingredient that was last approved up till Dec, 2024, when is its patent expiration date?\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\nReturn only the date only the year YYYY format of date: YYYY."
    else:
         prompt = f"First, for clinical trial {nct} among the {eff_group} effective regimen ingredients, identify which ingredient starts with the letter {first_letter}.\n\nThen, for this identified ingredient that was last approved up till Dec, 2024, when is its patent expiration date?\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\nReturn only the date only the year YYYY format of date: YYYY."

    # Gold standard should be the value from the column, even if the prompt asks for YYYY only.
    gold = patent_date
    return {"prompt": prompt, "gold": gold, "task_name": task_name}


def generate_exclusivity_prompt(row):
    task_name = "Exclusivity_Date"
    nct = row.get('NCT', 'UNKNOWN_NCT')
    eff_group = row.get('effecacy_group', 'UNKNOWN_EFFICACY_GROUP')
    ingredient = row.get('Ingredient', 'UNKNOWN_INGREDIENT')
    exclusivity_date = row.get('Exclusivity_Date') # Use get, allows None

    if pd.isna(ingredient) or ingredient == 'UNKNOWN_INGREDIENT':
         logging.warning(f"Missing 'Ingredient' for NCT {nct}. Skipping Exclusivity prompt generation.")
         return None

    first_letter = ingredient[0] if isinstance(ingredient, str) and len(ingredient) > 0 else '?'

    # Check for NA using pandas isna and also check if it's an empty string after stripping
    if pd.isna(exclusivity_date) or (isinstance(exclusivity_date, str) and exclusivity_date.strip() == ''):
        prompt = f"For clinical trial {nct} among the {eff_group} effective regimen ingredients, identify which ingredient starts with the letter {first_letter}.\n\nThen, for this identified ingredient that was last approved up till Dec, 2024, when is its exclusivity date according to the FDA?\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\nIf no exclusivity date exists or if it's not available, respond with 'DATE: NA'. Otherwise, return only the date in MM-DD-YYYY format of date: MM-DD-YYYY."
        gold = "NA"
    else:
        prompt = f"For clinical trial {nct} among the {eff_group} effective regimen ingredients, identify which ingredient starts with the letter {first_letter}.\n\nThen, for this identified ingredient that was last approved up till Dec, 2024, when is its exclusivity date according to the FDA?\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\nIf no exclusivity date exists or if it's not available, respond with 'DATE: NA'. Otherwise, return only the date in MM-DD-YYYY format of date: MM-DD-YYYY."
        gold = exclusivity_date # Use the value from the column
    return {"prompt": prompt, "gold": gold, "task_name": task_name}

def generate_approval_prompt_dynamic(row, gold_col):
    task_name = "Open_on_Approval"
    nct = row.get('NCT', 'UNKNOWN_NCT')
    eff_group = row.get('effecacy_group', 'UNKNOWN_EFFICACY_GROUP')
    ingredient = row.get('Ingredient', 'UNKNOWN_INGREDIENT')

    if pd.isna(ingredient) or ingredient == 'UNKNOWN_INGREDIENT':
         logging.warning(f"Missing 'Ingredient' for NCT {nct}. Skipping Approval prompt generation.")
         return None

    # Check if the specified gold column exists
    if gold_col not in row:
         logging.error(f"Missing specified gold standard column '{gold_col}' for Open_on_Approval task in input row for NCT {nct}. Skipping prompt generation.")
         # Consider raising an error or returning None depending on desired behavior
         # raise ValueError(f"Missing expected gold standard column '{gold_col}' for Open_on_Approval task in input row: {row.to_dict()}")
         return None # Skip this prompt if gold column is missing

    gold = row[gold_col] # Use the specified column for gold standard
    if pd.isna(gold):
        logging.warning(f"Missing gold value in column '{gold_col}' for NCT {nct}. Using NA as gold.")
        gold = "NA" # Or handle as appropriate, maybe skip?

    first_letter = ingredient[0] if isinstance(ingredient, str) and len(ingredient) > 0 else '?'

    if ingredient == 'APALUTAMIDE':
        prompt = f"For clinical trial {nct}. Among the {eff_group} effective regimen ingredients; find which is the ingredient with the first letter start with AP.\n\nThen, determine which pharmaceutical company received the most recent FDA approval (up until December 2024) for this identified ingredient.\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\n\nIf this company is listed on any US stock market (including over-the-counter markets), provide:\n1. The stock ticker symbol\n2. The opening stock price on the FDA approval date\n\nIf the company is not listed on any US stock market, please indicate \"NOT LISTED\"."
    else:
        prompt = f"For clinical trial {nct}, review the {eff_group} effective regimen ingredients and identify which ingredient starts with the letter {first_letter}.\n\nThen, determine which pharmaceutical company received the most recent FDA approval (up until December 2024) for this identified ingredient.\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\n\nIf this company is listed on any US stock market (including over-the-counter markets), provide:\n1. The stock ticker symbol\n2. The opening stock price on the FDA approval date\n\nIf the company is not listed on any US stock market, please indicate \"NOT LISTED\"."

    return {"prompt": prompt, "gold": gold, "task_name": task_name}


def main(input_csv, output_csv, gold_column_approval):
    logging.info(f"Reading input CSV: {input_csv}")
    try:
        # Read CSV, keeping empty strings as they are initially
        df_input = pd.read_csv(input_csv, keep_default_na=False)
        logging.info(f"Read {len(df_input)} rows from input file.")

        # Simple validation for required columns (using .get() in functions provides more robustness)
        base_required_cols = ['NCT', 'effecacy_group', 'Ingredient', 'Applicant_Full_Name', 'Patent_Expire_Date_Text', 'Exclusivity_Date']
        required_cols = base_required_cols + [gold_column_approval]
        missing = [col for col in required_cols if col not in df_input.columns]
        if missing:
             logging.error(f"Input CSV missing required columns: {missing}")
             # Decide whether to raise error or just warn and continue
             # For now, let's raise an error as these columns are crucial
             raise ValueError(f"Input CSV missing required columns: {missing}")

    except FileNotFoundError:
        logging.error(f"Input file not found: {input_csv}")
        return
    except ValueError as ve: # Catch specific validation error
        logging.error(f"Column validation failed: {ve}")
        return
    except Exception as e:
        logging.error(f"Error reading input CSV: {e}")
        traceback.print_exc()
        return

    output_data = []
    logging.info("Processing rows and generating prompts...")

    # Use iterrows for row-by-row processing
    for index, row in df_input.iterrows():
        if index % 100 == 0 and index > 0: # Log progress every 100 rows
             logging.info(f"Processing row {index}...")
        try:
            prompts_to_add = [
                generate_ingredient_prompt(row),
                generate_applicant_prompt(row),
                generate_patent_prompt(row),
                generate_exclusivity_prompt(row),
                generate_approval_prompt_dynamic(row, gold_column_approval) # Pass the gold column name
            ]
            # Filter out None results (rows skipped due to missing data)
            output_data.extend([p for p in prompts_to_add if p is not None])

        except Exception as e:
            logging.error(f"Error processing row {index}: {e} - Row data snippet: {row.to_dict().items()[:3]}...") # Log snippet
            traceback.print_exc() # Print full traceback for debugging
            # Optionally skip row or handle error differently

    logging.info(f"Generated {len(output_data)} rows for the output file.")

    if not output_data:
        logging.warning("No data generated. Output file will be empty or not created.")
        return

    df_output = pd.DataFrame(output_data)

    logging.info(f"Saving formatted data to: {output_csv}")
    try:
        # Ensure output directory exists if specified in the path
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")

        # Save with double quotes around all fields to handle commas/newlines in prompts
        df_output.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
        logging.info("Successfully saved output file.")
    except Exception as e:
        logging.error(f"Error writing output CSV: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format NCT 121 dataset similar to final50.csv. Creates 5 rows in output for each input row, one for each task type.")
    parser.add_argument("input_csv", help="Path to the input CSV file (e.g., data/final_nct_121_with_effecacy_group.csv)")
    parser.add_argument("output_csv", help="Path to save the formatted output CSV file (e.g., data/final121_formatted.csv)")
    parser.add_argument("--approval_gold_col", required=True, help="Name of the column in the input CSV containing the gold standard value (price or 'NOT LISTED') for the 'Open_on_Approval' task.")

    args = parser.parse_args()
    main(args.input_csv, args.output_csv, args.approval_gold_col)
