import pandas as pd
import argparse
import os
import logging
import json
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CSVGenerator")

# Define task types and their corresponding prompt templates
TASK_TEMPLATES = {
    "track_trial_ids": {
        "prompt_template": "Find/search the clinical trial id{context}\nOutput it in the format NCT<Number>",
        "expected_answer_column": "NCT",
        "required_columns": ["question 1", "NCT"]
    },
    "track_second_authors": {
        "prompt_template": "Find/search the second author of the paper{context}\nOutput it in the format SA<Second Author>",
        "expected_answer_column": "authors",
        "expected_answer_transform": lambda x: json.loads(x).get(list(json.loads(x).keys())[0], "").split("|")[1] if "|" in json.loads(x).get(list(json.loads(x).keys())[0], "") else "",
        "required_columns": ["question 1", "authors"]
    },
    "track_pmids": {
        "prompt_template": "Find/search the pubmed id of the paper{context}\nOutput it in the format pmid<pubmed id>",
        "expected_answer_column": "pmids",
        "required_columns": ["question 1", "pmids"]
    },
    "track_start_date": {
        "prompt_template": "Find/search the start date of the clinical trial {context}\n\nIMPORTANT: Respond with ONLY the format \"Start date: YYYY-MM\" or \"Start date: Month YYYY\". Do not include any other text in your response.",
        "expected_answer_column": "start_date",
        "required_columns": ["question 1", "start_date"]
    },
    "track_primary_outcomes": {
        "prompt_template": "Determine if the clinical trial {context} has primary outcomes.\n\nIMPORTANT: Respond with ONLY \"Primary outcomes: Yes\" or \"Primary outcomes: No\". Do not include any other text in your response.",
        "expected_answer_column": "has_primary_outcome",
        "required_columns": ["question 1", "has_primary_outcome"]
    },
    "track_secondary_outcomes": {
        "prompt_template": "Determine if the clinical trial {context} has secondary outcomes.\n\nIMPORTANT: Respond with ONLY \"Secondary outcomes: Yes\" or \"Secondary outcomes: No\". Do not include any other text in your response.",
        "expected_answer_column": "has_secondary_outcome",
        "required_columns": ["question 1", "has_secondary_outcome"]
    },
    "track_drug_route": {
        "prompt_template": "Find/search the route of administration for the drug used in the clinical trial {context}\n\nIMPORTANT: Respond with ONLY the format \"Drug route: X\" where X is the route (oral, intravenous, etc). If unknown, respond with \"Drug route: Unknown\". Do not include any other text in your response.",
        "expected_answer_column": "drug_routes",
        "required_columns": ["question 1", "drug_routes"]
    },
    "track_drug_class": {
        "prompt_template": "Find/search the drug class used in the clinical trial {context}\n\nIMPORTANT: Respond with ONLY the format \"Drug class: X\" where X is the class (antibiotics, chemotherapy, etc). Do not include any other text in your response.",
        "expected_answer_column": "drug_classes",
        "required_columns": ["question 1", "drug_classes"]
    },
    "track_second_authors_multiple_pmids": {
        "prompt_template": "please only return the US stock ticker (or OTC ticker if applicable) of {company} in all CAPS in the format of STOCK_TICKER: ticker else return NOT_LISTED",
        "expected_answer_column": "ticker",
        "required_columns": ["Applicant_Full_Name", "ticker"]
    },
    "Ingredient": {
        "prompt_template": "For clinical trial {nct}. Among the {efficacy_group} effective regimen ingredients; find which is the ingredient with the first letter start with {first_letter}\n return only the name of the ingredient or unknown in all CAPS in the format of INGREDIENT: name",
        "expected_answer_column": "Ingredient",
        "required_columns": ["NCT", "effecacy_group", "Ingredient"]
    },
    "Applicant_Full_Name": {
        "prompt_template": "First, for clinical trial {nct} among the {efficacy_group} effective regimen ingredients, identify which ingredient starts with the letter {first_letter}.\n\nThen, find which company has the latest FDA approval date up till Dec, 2024 for this identified ingredient. Note that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals. \nReturn only the company name in format of COMPANY: name.",
        "expected_answer_column": "Applicant_Full_Name",
        "required_columns": ["NCT", "effecacy_group", "Ingredient", "Applicant_Full_Name"]
    },
    "Patent_Expire_Date_Text": {
        "prompt_template": "First, for clinical trial {nct} among the {efficacy_group} effective regimen ingredients, identify which ingredient starts with the letter {first_letter}.\n\nThen, for this identified ingredient that was last approved up till Dec, 2024, when is its patent expiration date?\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\nReturn only the date only the year YYYY format of date: YYYY.",
        "expected_answer_column": "Patent_Expire_Date_Text",
        "required_columns": ["NCT", "effecacy_group", "Ingredient", "Patent_Expire_Date_Text"]
    },
    "Exclusivity_Date": {
        "prompt_template": "For clinical trial {nct} among the {efficacy_group} effective regimen ingredients, identify which ingredient starts with the letter {first_letter}.\n\nThen, for this identified ingredient that was last approved up till Dec, 2024, when is its exclusivity date according to the FDA?\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\nIf no exclusivity date exists or if it's not available, respond with 'DATE: NA'. Otherwise, return only the date in MM-DD-YYYY format of date: MM-DD-YYYY.",
        "expected_answer_column": "Exclusivity_Date",
        "required_columns": ["NCT", "effecacy_group", "Ingredient", "Exclusivity_Date"]
    },
    "Open_on_Approval": {
        "prompt_template": "For clinical trial {nct}, review the {efficacy_group} effective regimen ingredients and identify which ingredient starts with the letter {first_letter}.\n\nThen, determine which pharmaceutical company received the most recent FDA approval (up until December 2024) for this identified ingredient.\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\n\nIf this company is listed on any US stock market (including over-the-counter markets), provide:\n1. The stock ticker symbol\n2. The opening stock price on the FDA approval date\n\nIf the company is not listed on any US stock market, please indicate \"NOT LISTED\".",
        "expected_answer_column": "ticker",
        "expected_answer_transform": lambda x, row: f"{x}: ${float(row.get('Open_on_Approval', 0)):.2f}" if x != "NOT_LISTED" and x != "Not_Listed" else "NOT LISTED",
        "required_columns": ["NCT", "effecacy_group", "Ingredient", "ticker", "Open_on_Approval"],
        "use_ticker_format": True
    },
    "filled50": {
        "prompt_template": "{prompt}", # Direct copy of prompt
        "expected_answer_column": "gold",
        "required_columns": ["prompt", "gold"]
    },
    "filled121": {
        "prompt_template": "{prompt}", # Direct copy of prompt 
        "expected_answer_column": "gold",
        "required_columns": ["prompt", "gold"]
    },
    # Add additional task types for the deep research tasks
    "regime_drug_class": {
        "prompt_template": "For clinical trial {nct}, identify the class of drugs being used in the trial.\nReturn only the drug class in the format \"DRUG CLASS: X\" where X is the class of drugs.",
        "expected_answer_column": "drug_classes",
        "required_columns": ["NCT", "drug_classes"]
    },
    "latest_company_approval": {
        "prompt_template": "Which company has the latest FDA approval for {ingredient}?\nReturn only the company name in format of COMPANY: name.",
        "expected_answer_column": "Applicant_Full_Name",
        "required_columns": ["Ingredient", "Applicant_Full_Name"]
    },
    "patent_expiration_date": {
        "prompt_template": "When is the patent expiration date for {ingredient} that was approved by the FDA?\nReturn only the year in YYYY format as DATE: YYYY.",
        "expected_answer_column": "Patent_Expire_Date_Text",
        "required_columns": ["Ingredient", "Patent_Expire_Date_Text"]
    },
    "exclusivity_Date": {
        "prompt_template": "When is the exclusivity date for {ingredient} according to the FDA?\nIf no exclusivity date exists, respond with 'DATE: NA'. Otherwise, return only the date in MM-DD-YYYY format as DATE: MM-DD-YYYY.",
        "expected_answer_column": "Exclusivity_Date",
        "required_columns": ["Ingredient", "Exclusivity_Date"]
    },
    "ceo_name": {
        "prompt_template": "Who is the current CEO of {company}?\nReturn only the name in the format CEO: [Name].",
        "expected_answer_column": "ceo_name", # This would need to be added to your dataset
        "required_columns": ["Applicant_Full_Name"],
        "needs_external_data": True
    }
}

def create_task_csv(input_csv, task_name, output_dir, num_examples=None):
    """
    Create a CSV file for a specific task with the proper prompt and expected answer
    
    Args:
        input_csv: Path to input CSV file
        task_name: Name of the task
        output_dir: Directory to save the output CSV
        num_examples: Optional limit on number of examples to generate
        
    Returns:
        Path to the created CSV file
    """
    if task_name not in TASK_TEMPLATES:
        logger.error(f"Unknown task: {task_name}")
        return None
    
    template = TASK_TEMPLATES[task_name]
    required_columns = template["required_columns"]
    
    try:
        # Read input CSV
        logger.info(f"Reading input CSV: {input_csv}")
        df = pd.read_csv(input_csv, low_memory=False)
        
        # Handle JSON string columns - try to parse any column that might be JSON
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Check first non-null value to see if it looks like JSON
                    first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if first_val and isinstance(first_val, str) and first_val.startswith('{') and first_val.endswith('}'):
                        logger.info(f"Column {col} appears to be JSON, attempting to parse")
                        # We don't actually convert it yet, just note that it's JSON
                except (IndexError, AttributeError):
                    pass
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns for task {task_name}: {missing_columns}")
            return None
        
        # Filter out rows with missing required data
        for col in required_columns:
            df = df[~df[col].isna()]
        
        if num_examples and num_examples < len(df):
            logger.info(f"Limiting to {num_examples} examples")
            df = df.head(num_examples)
        
        # Create a new DataFrame for the task
        task_data = []
        
        logger.info(f"Processing {len(df)} rows for task: {task_name}")
        for i, row in df.iterrows():
            row_dict = row.to_dict()
            
            # Format the prompt based on the task
            try:
                if task_name in ["track_trial_ids", "track_second_authors", "track_pmids", 
                              "track_start_date", "track_primary_outcomes", "track_secondary_outcomes",
                              "track_drug_route", "track_drug_class"]:
                    # Extract context from question
                    context = row['question 1'].split('Choose an option')[1] if 'Choose an option' in str(row['question 1']) else row['question 1']
                    prompt = template["prompt_template"].format(context=context)
                elif task_name in ["track_second_authors_multiple_pmids"]:
                    prompt = template["prompt_template"].format(company=row["Applicant_Full_Name"])
                elif task_name in ["Ingredient", "Applicant_Full_Name", "Patent_Expire_Date_Text", 
                                 "Exclusivity_Date", "Open_on_Approval"]:
                    # Special handling for the deep research tasks
                    first_letter = row["Ingredient"][0] if row["Ingredient"] != "APALUTAMIDE" else "AP"
                    prompt = template["prompt_template"].format(
                        nct=row["NCT"], 
                        efficacy_group=row["effecacy_group"] if "effecacy_group" in row else "more",
                        first_letter=first_letter
                    )
                elif task_name in ["regime_drug_class"]:
                    prompt = template["prompt_template"].format(nct=row["NCT"])
                elif task_name in ["latest_company_approval", "patent_expiration_date", "exclusivity_Date"]:
                    prompt = template["prompt_template"].format(ingredient=row["Ingredient"])
                elif task_name in ["ceo_name"]:
                    prompt = template["prompt_template"].format(company=row["Applicant_Full_Name"])
                elif task_name in ["filled50", "filled121"]:
                    # For these tasks, we need a 'prompt' column in the input data
                    if "prompt" not in row:
                        logger.warning(f"Row {i} missing 'prompt' column required for {task_name}")
                        continue
                    prompt = template["prompt_template"].format(prompt=row["prompt"])
                else:
                    logger.warning(f"Unsupported task format: {task_name}, skipping row {i}")
                    continue
            except KeyError as e:
                logger.warning(f"Missing key {e} for row {i} in task {task_name}, skipping")
                continue
            except Exception as e:
                logger.warning(f"Error formatting prompt for row {i} in task {task_name}: {e}")
                continue
                
            # Get the expected answer
            try:
                if "expected_answer_transform" in template:
                    # Some transforms need the whole row
                    if template["expected_answer_transform"].__code__.co_argcount > 1:
                        expected_answer = template["expected_answer_transform"](
                            row[template["expected_answer_column"]], row_dict)
                    else:
                        expected_answer = template["expected_answer_transform"](
                            row[template["expected_answer_column"]])
                else:
                    expected_answer = row[template["expected_answer_column"]]
                    
                # Handle special cases
                if task_name == "Exclusivity_Date" or task_name == "exclusivity_Date":
                    if pd.isna(expected_answer) or expected_answer is None or str(expected_answer).strip() == '':
                        expected_answer = "NA"
                elif task_name == "Open_on_Approval" and "use_ticker_format" in template and template["use_ticker_format"]:
                    # For stock tickers, format as TICKER: $PRICE
                    ticker = row.get("ticker", "NOT_LISTED")
                    price = row.get("Open_on_Approval", "0.00")
                    
                    if ticker in ["NOT_LISTED", "Not_Listed"] or pd.isna(ticker):
                        expected_answer = "NOT LISTED"
                    else:
                        try:
                            # Format the price as $XX.XX
                            if not pd.isna(price):
                                price_float = float(price)
                                expected_answer = f"{ticker}: ${price_float:.2f}"
                            else:
                                expected_answer = f"{ticker}: $0.00"
                        except (ValueError, TypeError):
                            expected_answer = f"{ticker}: ${price}"
                elif task_name == "track_second_authors":
                    # Handle the JSON format for authors
                    authors_json = row["authors"]
                    try:
                        authors_dict = json.loads(authors_json)
                        # Get the first PMID's authors
                        first_pmid = list(authors_dict.keys())[0]
                        authors_list = authors_dict[first_pmid].split('|')
                        if len(authors_list) >= 2:
                            expected_answer = authors_list[1]
                        else:
                            logger.warning(f"No second author found for row {i}")
                            expected_answer = ""
                    except (json.JSONDecodeError, IndexError, KeyError) as e:
                        logger.warning(f"Error parsing authors for row {i}: {e}")
                        expected_answer = ""
                        
            except KeyError as e:
                logger.warning(f"Missing key {e} for row {i} in task {task_name}, using default")
                expected_answer = "NA" if task_name in ["Exclusivity_Date", "exclusivity_Date"] else ""
            except Exception as e:
                logger.warning(f"Error getting expected answer for row {i} in task {task_name}: {e}")
                expected_answer = ""
                
            # Create entry for this row with only the required columns
            task_entry = {
                "prompt": prompt,
                "expected_answer": expected_answer,
            }
            
            # Add only the required columns from the original data
            for col in required_columns:
                if col in row:
                    task_entry[col] = row[col]
            
            # Add the task name to identify the task
            task_entry["task_name"] = task_name
            
            task_data.append(task_entry)
            
        if not task_data:
            logger.warning(f"No valid entries found for task {task_name}")
            return None
            
        # Create the output dataframe
        task_df = pd.DataFrame(task_data)
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the output CSV
        output_path = os.path.join(output_dir, f"{task_name}.csv")
        task_df.to_csv(output_path, index=False)
        
        logger.info(f"Created CSV for task {task_name} at {output_path} with {len(task_df)} entries")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating CSV for task {task_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate CSV files for each task type")
    parser.add_argument("input_csv", help="Path to input CSV file with all required columns")
    parser.add_argument("--tasks", nargs="+", 
                        choices=list(TASK_TEMPLATES.keys()),
                        default=["Ingredient", "Applicant_Full_Name", "Patent_Expire_Date_Text", "Exclusivity_Date", "Open_on_Approval"],
                        help="Tasks to generate CSVs for (default: core research tasks)")
    parser.add_argument("--output_dir", default="task_csvs", help="Directory to save output CSVs")
    parser.add_argument("--num_examples", type=int, help="Limit the number of examples per task (optional)")
    parser.add_argument("--combine", action="store_true", help="Create a combined CSV with all tasks")
    
    args = parser.parse_args()
    
    logger.info(f"Generating CSVs from {args.input_csv} for tasks: {args.tasks}")
    
    # Check if input file exists
    if not os.path.exists(args.input_csv):
        logger.error(f"Input file does not exist: {args.input_csv}")
        return

    # Dictionary to store task data for combined CSV
    all_tasks_data = []
    
    # Process each task
    for task in args.tasks:
        output_path = create_task_csv(
            args.input_csv, 
            task, 
            args.output_dir,
            num_examples=args.num_examples
        )
        
        if output_path and args.combine:
            # Read the task CSV and add to combined data
            try:
                task_df = pd.read_csv(output_path)
                for _, row in task_df.iterrows():
                    all_tasks_data.append(row.to_dict())
            except Exception as e:
                logger.error(f"Error reading task CSV for combining: {e}")
    
    # Create combined CSV if requested
    if args.combine and all_tasks_data:
        combined_path = os.path.join(args.output_dir, "all_tasks_combined.csv")
        try:
            combined_df = pd.DataFrame(all_tasks_data)
            combined_df.to_csv(combined_path, index=False)
            logger.info(f"Created combined CSV at {combined_path} with {len(combined_df)} entries")
        except Exception as e:
            logger.error(f"Error creating combined CSV: {e}")
    
    logger.info("CSV generation complete")

if __name__ == "__main__":
    main()