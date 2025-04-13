import csv
import os
import re
import pandas as pd
import json
import argparse
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from gemini_inference import GeminiInference, run_inference_multithread, GEMINI_MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nct_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NCTProcessor")

def extract_from_response(response: str, task: str = "track_trial_ids") -> str:
    """
    Extract the relevant information from model response based on task
    
    Args:
        response: Model response text
        task: Task name to determine extraction pattern
        
    Returns:
        Extracted information or empty string if not found
    """
    # Add debugging - log the response
    #logger.debug(f"Extracting info from response (task: {task}):\n{response[:500]}...")
    logger.info(f"Extracting info from response (task: {task}):\n{response[:800]}...")
    if task == "track_trial_ids":
        # Look for the pattern NCT followed by 8 digits
        match = re.search(r'NCT\d{8}', response)
        if match:
            return match.group(0)
        return ""
    elif task == "track_second_authors":
        # Look for the pattern SA followed by any text
        match = re.search(r'(?i)SA[:\s]*[<]?([^>]+)[>]?', response)
        if match:
            return match.group(1).strip()
        return ""
    elif task == "track_pmids":
        # Look for the pattern pmid followed by any text
        match = re.search(r'(?i)pmid[:\s]*(\d+)', response)
        if match:
            return match.group(1)
        return ""
    elif task == "track_second_authors_multiple_pmids":
        # Try more flexible pattern matching for PMID and second author
        
        # Various ways the model might output a PMID
        pmid_patterns = [
            r'(?i)\*\*\s*pmid\s*:\s*\*\*\s*(\d+)',  # **PMID:** 12345
            r'(?i)pmid[:\s]*(\d+)',                           # PMID: 12345
            r'(?i)pubmed\s+id[:\s]*(\d+)',                    # PubMed ID: 12345
            r'(?i)pubmed\s+identifier[:\s]*(\d+)',            # PubMed Identifier: 12345
            r'(?i)\bpmid\b[^0-9]*(\d+)',                      # PMID 12345
            r'(?i)\[(\d+)\]',                                 # [12345]
            r'(?i)pubmed[:\s]*(\d+)'                          # PubMed: 12345
        ]
        
        # Various ways the model might output a second author
        author_patterns = [
            r'(?i)\*\*\s*SA\s*:\s*\*\*\s*([A-Za-z\s\.,\-\']+?)(?:$|\n|\.|,)',  # **SA:** Smith, J.
            r'(?i)second\s+author[:\s]*([A-Za-z\s\.,\-\']+?)(?:$|\n|\.|,)',   # Second author: Smith, J.
            r'(?i)SA[:\s]*([A-Za-z\s\.,\-\']+?)(?:$|\n|\.|,)',                # SA: Smith, J.
            r'(?i)2nd\s+author[:\s]*([A-Za-z\s\.,\-\']+?)(?:$|\n|\.|,)',      # 2nd author: Smith, J.
            r'(?i)author\s*#?\s*2[:\s]*([A-Za-z\s\.,\-\']+?)(?:$|\n|\.|,)',   # Author #2: Smith, J.
            r'(?i)second\s+author\s+is\s+([A-Za-z\s\.,\-\']+?)(?:$|\n|\.|,)'  # Second author is Smith, J.
        ]
        
        # Try each PMID pattern
        pmid = None
        for pattern in pmid_patterns:
            match = re.search(pattern, response)
            if match:
                pmid = match.group(1).strip()
                logger.debug(f"Found PMID using pattern {pattern}: {pmid}")
                break
        
        # Try each author pattern
        author = None
        for pattern in author_patterns:
            match = re.search(pattern, response)
            if match:
                author = match.group(1).strip()
                logger.debug(f"Found author using pattern {pattern}: {author}")
                break
        
        # If we found both PMID and author, return them
        if pmid and author:
            logger.info(f"Successfully extracted PMID: {pmid} and Author: {author}")
            return f"{pmid}|{author}"
        else:
            # If not found using patterns, try a more general approach
            # Just try to find any PMID-like number and any name-like text after "second author" or similar
            general_pmid_match = re.search(r'(?i)(?:pmid|pubmed)[^0-9]*(\d+)', response)
            general_author_match = re.search(r'(?i)(?:second author|2nd author|author 2)(?:[:\s]*)([A-Za-z][A-Za-z\s\.,\-\']{2,})', response)
            
            if general_pmid_match and general_author_match:
                pmid = general_pmid_match.group(1).strip()
                author = general_author_match.group(1).strip()
                logger.info(f"Extracted using general patterns - PMID: {pmid}, Author: {author}")
                return f"{pmid}|{author}"
            
            # Still not found, log the failure
            logger.warning(f"Failed to extract PMID and author. PMID found: {pmid is not None}, Author found: {author is not None}")
            
            # Extra debugging to understand the model's output format
            logger.debug(f"Model output for failed extraction:\n{response[:1000]}")
            return ""

    elif task == "track_second_authors_multiple_pmids_any":
        # Just extract the second author, any PMID is fine
        # Similar improvements as above
        author_patterns = [
            r'(?i)\*\*\s*SA\s*:\s*\*\*\s*([A-Za-z\s\.,\-\']+?)(?:$|\n|\.|,)',  # **SA:** Smith, J.
            r'(?i)second\s+author[:\s]*([A-Za-z\s\.,\-\']+?)(?:$|\n|\.|,)',   # Second author: Smith, J.
            r'(?i)SA[:\s]*([A-Za-z\s\.,\-\']+?)(?:$|\n|\.|,)',                # SA: Smith, J.
            r'(?i)2nd\s+author[:\s]*([A-Za-z\s\.,\-\']+?)(?:$|\n|\.|,)',      # 2nd author: Smith, J.
            r'(?i)author\s*#?\s*2[:\s]*([A-Za-z\s\.,\-\']+?)(?:$|\n|\.|,)',   # Author #2: Smith, J.
            r'(?i)second\s+author\s+is\s+([A-Za-z\s\.,\-\']+?)(?:$|\n|\.|,)'  # Second author is Smith, J.
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, response)
            if match:
                author = match.group(1).strip()
                logger.debug(f"Found author using pattern {pattern}: {author}")
                return author
        
        # General fallback
        general_match = re.search(r'(?i)(?:second author|2nd author|author 2)(?:[:\s]*)([A-Za-z][A-Za-z\s\.,\-\']{2,})', response)
        if general_match:
            author = general_match.group(1).strip()
            logger.debug(f"Found author using general pattern: {author}")
            return author
            
        logger.warning(f"Failed to extract any author.")
        return ""
    
    elif task == "track_start_date":
        # Single pattern to match "Start date: YYYY-MM" format
        direct_pattern = r'(?i)start\s*date:?\s*((?:\d{4}-\d{2})|(?:[A-Za-z]+\s+\d{4}))'
        match = re.search(direct_pattern, response)
        if match:
            return match.group(1).strip()
        
        # More general patterns if the specific format wasn't found
        general_patterns = [
            r'(?i)(?:start|begin|commence)(?:\w*)?\s*(?:date|on):?\s*((?:\d{4}-\d{2})|(?:[A-Za-z]+\s+\d{4})|\d{4})',
            r'(?i)date:?\s*((?:\d{4}-\d{2})|(?:[A-Za-z]+\s+\d{4})|\d{4})',
            r'(?i)(\d{4}-\d{2})',  # Just look for YYYY-MM format anywhere
            r'(?i)((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})'
        ]
        
        for pattern in general_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
                
        logger.warning("Failed to extract start date.")
        return ""
        
    elif task == "track_primary_outcomes":
        # Look for exact "Primary outcomes: Yes/No" format first
        direct_pattern = r'(?i)primary\s*outcomes?:?\s*(yes|no)\b'
        match = re.search(direct_pattern, response)
        if match:
            return match.group(1).capitalize()
        
        # Alternative formats
        yes_pattern = r'(?i)(?:has|with|contains)\s+primary\s+outcomes?:?\s*(?:yes|true|present)'
        no_pattern = r'(?i)(?:no|without|lacks)\s+primary\s+outcomes?'
        
        if re.search(yes_pattern, response):
            return "Yes"
        if re.search(no_pattern, response):
            return "No"
        
        # Fall back to simple Yes/No search
        if re.search(r'(?i)\byes\b', response):
            return "Yes"
        if re.search(r'(?i)\bno\b', response):
            return "No"
            
        logger.warning("Failed to extract primary outcomes information.")
        return ""
        
    elif task == "track_secondary_outcomes":
        # Look for exact "Secondary outcomes: Yes/No" format first
        direct_pattern = r'(?i)secondary\s*outcomes?:?\s*(yes|no)\b'
        match = re.search(direct_pattern, response)
        if match:
            return match.group(1).capitalize()
        
        # Alternative formats
        yes_pattern = r'(?i)(?:has|with|contains)\s+secondary\s+outcomes?:?\s*(?:yes|true|present)'
        no_pattern = r'(?i)(?:no|without|lacks)\s+secondary\s+outcomes?'
        
        if re.search(yes_pattern, response):
            return "Yes"
        if re.search(no_pattern, response):
            return "No"
        
        # Fall back to simple Yes/No search
        if re.search(r'(?i)\byes\b', response):
            return "Yes"
        if re.search(r'(?i)\bno\b', response):
            return "No"
            
        logger.warning("Failed to extract secondary outcomes information.")
        return ""
    else:
        logger.error(f"Unknown task: {task}")
        return ""
    
def get_second_author(authors_json: str, pmid: str = None) -> Union[str, Dict[str, str]]:
    """
    Get the second author for a specific PMID or all second authors
    
    Args:
        authors_json: JSON string with PMID to authors mapping
        pmid: Optional specific PMID to get second author for
        
    Returns:
        Second author string if pmid is specified, otherwise dictionary of PMID to second author
    """
    try:
        # Parse the JSON string
        import json
        authors_dict = json.loads(authors_json)
        
        if pmid:
            # Get second author for specific PMID
            if pmid in authors_dict:
                authors_list = authors_dict[pmid].split('|')
                if len(authors_list) >= 2:
                    return authors_list[1]
                else:
                    return "NO_SECOND_AUTHOR"
            else:
                return "PMID_NOT_FOUND"
        else:
            # Get all second authors
            second_authors = {}
            for pid, authors in authors_dict.items():
                authors_list = authors.split('|')
                if len(authors_list) >= 2:
                    second_authors[pid] = authors_list[1]
            return second_authors
    except Exception as e:
        logger.error(f"Error parsing authors JSON: {str(e)}")
        return {} if pmid is None else "ERROR"

def process_nct_csv(
    csv_path: str,
    model_name: str = "gemini-2.0-flash",
    use_tools: bool = False,
    max_workers: int = 4,
    output_path: str = None,
    test_mode: bool = False,
    n: int = None,
    task: str = None
) -> List[Dict]:
    """
    Process CSV file with NCT predictions
    
    Args:
        csv_path: Path to CSV file
        model_name: Gemini model to use
        use_tools: Whether to use Google Search tool
        max_workers: Number of parallel threads
        output_path: Optional path for saving results CSV
        test_mode: Whether to run in test mode (only 8 examples)
        n: Number of rows to process (for testing)
        task: Task to perform
        
    Returns:
        List of result dictionaries
    """
    try:
        # Read CSV file
        logger.info(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        debug_dir = "debug_search_results"
        if use_tools:
            os.makedirs(debug_dir, exist_ok=True)
        
        # If test mode, limit to 8 examples
        if test_mode:
            df = df.head(8)
            logger.info("TEST MODE: Limited to 8 examples")
        elif n is not None:
            df = df.head(n)
            logger.info(f"Processing first {n} examples")
        
        # Create list of inputs
        inputs = []
        rows = []
        answers = []

        logger.info("Preparing input data...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing inputs"):
            if task == "track_trial_ids":
                # track trial ids
                question = "Find/search the clinical trial id " + row['question 1'].split('Choose an option')[1] + '\nOutput it in the format NCT<Number>'
                correct_answer = row['NCT']
            elif task == "track_second_authors":
                # track second authors (old format)
                question = "Find/search the second author of the paper " + row['question 1'].split('Choose an option')[1] + '\nOutput it in the format SA<Second Author>'
                correct_answer = row['authors'].split('|')[1] if '|' in row['authors'] else ""
            elif task == "track_pmids":
                # track pmids
                question = "Find/search the pubmed id of the paper " + row['question 1'].split('Choose an option')[1] + '\nOutput it in the format pmid<pubmed id>'
                correct_answer = row['pmids']
            elif task == "track_start_date":
                # Track start date of the clinical trial with stricter format instructions
                question = "Find/search the start date of the clinical trial " + row['question 1'].split('Choose an option')[1] + '\n\nIMPORTANT: Respond with ONLY the format "Start date: YYYY-MM" or "Start date: Month YYYY". Do not include any other text in your response.'
                correct_answer = row['start_date']
            elif task == "track_primary_outcomes":
                # Track whether the trial has primary outcomes with stricter format
                question = "Determine if the clinical trial " + row['question 1'].split('Choose an option')[1] + ' has primary outcomes.\n\nIMPORTANT: Respond with ONLY "Primary outcomes: Yes" or "Primary outcomes: No". Do not include any other text in your response.'
                correct_answer = row['has_primary_outcome']
            elif task == "track_secondary_outcomes":
                # Track whether the trial has secondary outcomes with stricter format
                question = "Determine if the clinical trial " + row['question 1'].split('Choose an option')[1] + ' has secondary outcomes.\n\nIMPORTANT: Respond with ONLY "Secondary outcomes: Yes" or "Secondary outcomes: No". Do not include any other text in your response.'
                correct_answer = row['has_secondary_outcome']
            # In process_nct_csv function where task handling occurs:
            elif task == "track_second_authors_multiple_pmids":
                # Format: model needs to determine both PMID and second author
                question = "Find/search the pubmed ID and second author of the paper referenced in " + row['question 1'].split('Choose an option')[1] + '\nOutput it in the format PMID<pubmed id> SA<Second Author>'
                
                # For validation, we'll need all valid PMID-second author pairs
                pmid_author_pairs = {}
                try:
                    import json
                    authors_dict = json.loads(row['authors'])
                    for pmid, authors_str in authors_dict.items():
                        authors_list = authors_str.split('|')
                        if len(authors_list) >= 2:
                            pmid_author_pairs[pmid] = authors_list[1]
                except Exception as e:
                    logger.error(f"Error parsing authors JSON: {str(e)}")
                    
                # Create a simpler format for validation
                correct_pairs = []
                for pmid, author in pmid_author_pairs.items():
                    correct_pairs.append(f"{pmid}|{author}")
                correct_answer = "||".join(correct_pairs)
            elif task == "track_second_authors_multiple_pmids_any":
                # Format: find any second author from any of the PMIDs
                question = "Find/search the second author of any of the papers referenced in " + row['question 1'].split('Choose an option')[1] + '\nOutput it in the format SA<Second Author>'
                # Get all second authors
                second_authors_dict = get_second_author(row['authors'])
                # Join all second authors with a delimiter
                correct_answer = "|".join(second_authors_dict.values())
            else:
                logger.error(f"Unknown task: {task}")
                continue
            
            # Format the prompt
            prompt = question
            inputs.append(prompt)
            answers.append(correct_answer)
            rows.append(row)
        
        # Run inference in parallel
        logger.info(f"Processing {len(inputs)} examples using {model_name}" + 
              f" {'with' if use_tools else 'without'} tools" +
              f" using {max_workers} threads...")
        
        results = run_inference_multithread(
            model_name=model_name,
            input_list=inputs,
            use_tools=use_tools,
            max_workers=max_workers
        )
        
        # Process results
        processed_results = []
        correct_count = 0
        logger.info("Processing results...")

        for i, (result, answer, prompt, row) in enumerate(tqdm(zip(results, answers, inputs, rows), total=len(results), desc="Processing results")):
            if use_tools and i < 5:  # Save first 5 responses for debugging
                try:
                    import json
                    with open(f"{debug_dir}/raw_response_{i}.json", 'w') as f:
                        json.dump({
                            "response": result,
                        }, f, indent=2, default=str)
                except Exception as e:
                    logger.error(f"Error saving debug data: {str(e)}")
            
            try:
                # Extract response text and URLs based on the response format
                if isinstance(result, dict) and 'text' in result and 'citations' in result:
                    response_text = result['text']
                    urls = result['citations']
                else:
                    response_text = result
                    urls = []
                
                # Extract information from response based on task
                extracted_info = extract_from_response(response_text, task=task)
                
                # Check if the extracted information is correct
                is_correct = False
                
                if task == "track_trial_ids":
                    nct_list = [nct.strip() for nct in row['NCT'].split(',')]
                    is_correct = extracted_info in nct_list
                elif task == "track_second_authors":
                    # Old format - direct comparison
                    is_correct = extracted_info.strip().lower() == answer.strip().lower()
                elif task == "track_pmids":
                    pmid_list = [str(int(pmid)) for pmid in row['pmids'].split(',')]
                    is_correct = extracted_info in pmid_list
                elif task == "track_second_authors_multiple_pmids":
                    # Get all valid PMID|author pairs from the answer
                    valid_pairs = answer.split("||")
                    
                    # Format: PMID|SecondAuthor
                    if extracted_info and '|' in extracted_info:
                        extracted_pmid, extracted_author = extracted_info.split('|', 1)
                        extracted_pmid = extracted_pmid.strip()
                        extracted_author = extracted_author.strip().lower()
                        
                        # Check if any valid pair matches the extracted pair
                        is_correct = False
                        for pair in valid_pairs:
                            if '|' in pair:
                                valid_pmid, valid_author = pair.split('|', 1)
                                if extracted_pmid == valid_pmid.strip() and extracted_author == valid_author.strip().lower():
                                    is_correct = True
                                    break
                    else:
                        is_correct = False
                elif task == "track_second_authors_multiple_pmids_any":
                    # Check if extracted author matches any of the valid second authors
                    valid_authors = [author.strip().lower() for author in answer.split('|')]
                    is_correct = extracted_info.strip().lower() in valid_authors
                elif task == "track_start_date":
                    # Simpler date normalization
                    extracted_date = extracted_info.strip()
                    expected_date = answer.strip()
                    
                    # First try direct comparison
                    if extracted_date.lower() == expected_date.lower():
                        is_correct = True
                    else:
                        # Try to normalize formats
                        try:
                            # Extract year and month from both strings
                            expected_year = re.search(r'(\d{4})', expected_date).group(1) if re.search(r'(\d{4})', expected_date) else None
                            extracted_year = re.search(r'(\d{4})', extracted_date).group(1) if re.search(r'(\d{4})', extracted_date) else None
                            
                            # If we can extract years from both, compare them first
                            if expected_year and extracted_year and expected_year == extracted_year:
                                # If only checking year, consider it correct if no month info in expected
                                if re.match(r'^\d{4}$', expected_date):
                                    is_correct = True
                                else:
                                    # Check month matching
                                    month_map = {
                                        '01': ['jan', 'january'],
                                        '02': ['feb', 'february'],
                                        '03': ['mar', 'march'],
                                        '04': ['apr', 'april'],
                                        '05': ['may'],
                                        '06': ['jun', 'june'],
                                        '07': ['jul', 'july'],
                                        '08': ['aug', 'august'],
                                        '09': ['sep', 'september'],
                                        '10': ['oct', 'october'],
                                        '11': ['nov', 'november'],
                                        '12': ['dec', 'december']
                                    }
                                    
                                    # Extract month from expected date
                                    expected_month = None
                                    if re.search(r'-(\d{2})', expected_date):
                                        expected_month = re.search(r'-(\d{2})', expected_date).group(1)
                                    else:
                                        for num, names in month_map.items():
                                            if any(name in expected_date.lower() for name in names):
                                                expected_month = num
                                                break
                                    
                                    # Extract month from extracted date
                                    extracted_month = None
                                    if re.search(r'-(\d{2})', extracted_date):
                                        extracted_month = re.search(r'-(\d{2})', extracted_date).group(1)
                                    else:
                                        for num, names in month_map.items():
                                            if any(name in extracted_date.lower() for name in names):
                                                extracted_month = num
                                                break
                                    
                                    # Compare months if both are found
                                    is_correct = expected_month and extracted_month and expected_month == extracted_month
                            else:
                                is_correct = False
                                
                        except Exception as e:
                            logger.error(f"Error comparing dates: {str(e)}")
                            # Fallback to basic comparison
                            is_correct = extracted_date.lower() == expected_date.lower()

                elif task in ["track_primary_outcomes", "track_secondary_outcomes"]:
                    # Simple yes/no comparison, case-insensitive
                    is_correct = extracted_info.strip().lower() == answer.strip().lower()
                if is_correct:
                    correct_count += 1
                
                # Create result dictionary
                result_dict = {
                    'question': prompt,
                    'correct_answer': answer,
                    'model_output': response_text,
                    'extracted_info': extracted_info,
                    'correct': is_correct,
                    'urls': ', '.join(urls) if urls else ''
                }
                processed_results.append(result_dict)
                
            except Exception as e:
                logger.error(f"Error processing result {i}: {str(e)}")
                logger.error(traceback.format_exc())
                # Add a placeholder result with error information
                processed_results.append({
                    'question': prompt,
                    'correct_answer': answer,
                    'model_output': f"ERROR: {str(e)}",
                    'extracted_info': '',
                    'correct': False,
                    'urls': ''
                })

        # Calculate accuracy
        accuracy = correct_count / len(processed_results) if processed_results else 0
        logger.info(f"Accuracy: {correct_count}/{len(processed_results)} ({accuracy:.2%})")
        
        # Save results to CSV if output path provided
        if output_path:
            logger.info(f"Saving results to {output_path}...")
            output_df = pd.DataFrame(processed_results)
            output_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def main():
    parser = argparse.ArgumentParser(description="Evaluate model's ability to extract various information")
    parser.add_argument("csv_path", help="Path to CSV file with required columns")
    parser.add_argument("--model", default="gemini-2.0-flash", choices=list(GEMINI_MODELS.keys()), 
                        help="Gemini model to use")
    parser.add_argument("--use_tools", action="store_true", help="Use Google Search tool")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--output", help="Output CSV path (optional)")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 8 examples")
    parser.add_argument("-n", type=int, help="Process only the first n rows (for testing)")
    parser.add_argument("--debug", action="store_true", help="Save debug information")
    parser.add_argument("--task", 
                        choices=["track_trial_ids", "track_second_authors", "track_pmids",
                                "track_second_authors_multiple_pmids", 
                                "track_second_authors_multiple_pmids_any",
                                "track_start_date",
                                "track_primary_outcomes",
                                "track_secondary_outcomes"], 
                        default="track_trial_ids",
                        help="Task to perform",
                        )
    
    args = parser.parse_args()
    
    if args.model not in GEMINI_MODELS:
        logger.error(f"Invalid model name: {args.model}")
        logger.error(f"Available models: {', '.join(GEMINI_MODELS.keys())}")
        return
    
    logger.info(f"Processing {args.csv_path} with {args.model}")
    process_nct_csv(
        csv_path=args.csv_path,
        model_name=args.model,
        use_tools=args.use_tools,
        max_workers=args.threads,
        output_path=args.output,
        test_mode=args.test,
        n=args.n,
        task=args.task
    )

if __name__ == "__main__":
    main()