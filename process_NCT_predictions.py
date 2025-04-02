import csv
import os
import re
import pandas as pd
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

def extract_nct_from_response(response: str) -> str:
    """
    Extract the NCT number from model response
    
    Args:
        response: Model response text
        
    Returns:
        Extracted NCT number or empty string if not found
    """
    # Look for the pattern NCT followed by 8 digits
    match = re.search(r'NCT\d{8}', response)
    if match:
        return match.group(0)
    return ""

def format_nct_prompt(evidence: str) -> str:
    """
    Format the prompt for the model to extract NCT number
    
    Args:
        evidence: The evidence text
        
    Returns:
        Formatted prompt
    """
    prompt = f"""Search on the web the clinical trial id that showed this, start with NCT: {evidence}
    
Output it in the format NCT<Number>
"""
    return prompt

def process_nct_csv(
    csv_path: str,
    model_name: str = "gemini-2.0-flash",
    use_tools: bool = False,
    max_workers: int = 4,
    output_path: str = None,
    test_mode: bool = False,
    n: int = None
) -> List[Dict]:
    """
    Process CSV file with NCT predictions
    
    Args:
        csv_path: Path to CSV file with evidence and NCT columns
        model_name: Gemini model to use
        use_tools: Whether to use Google Search tool
        max_workers: Number of parallel threads
        output_path: Optional path for saving results CSV
        test_mode: Whether to run in test mode (only 8 examples)
        n: Number of rows to process (for testing)
        
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
        
        logger.info("Preparing input data...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing inputs"):
            evidence = row['evidence']
            correct_nct = row['NCT']
            
            # Format the prompt
            prompt = format_nct_prompt(evidence)
            inputs.append(prompt)
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
        
        # Add debug for results
        logger.info(f"Received {len(results)} results")
        logger.info(f"First result type: {type(results[0]) if results else 'No results'}")
        
        # Process results
        processed_results = []
        correct_count = 0
        logger.info("Processing results...")

        for i, (result, row) in enumerate(tqdm(zip(results, rows), total=len(results), desc="Processing results")):
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
                # Add debug info
                logger.info(f"Processing result {i}, type: {type(result)}")
                
                # Extract response text and URLs based on the new response format
                if isinstance(result, dict) and 'text' in result and 'citations' in result:
                    response_text = result['text']
                    urls = result['citations']
                else:
                    response_text = result
                    urls = []
                
                # Extract NCT from response
                extracted_nct = extract_nct_from_response(response_text)
                
                # Check if the extracted NCT is in the correct NCT (which might have multiple NCT numbers)
                is_correct = False
                nct_list = [nct.strip() for nct in row['NCT'].split(',')]
                if extracted_nct in nct_list:
                    is_correct = True
                    correct_count += 1
                
                # Create result dictionary with URLs
                result_dict = {
                    'evidence': row['evidence'],
                    'correct_nct': row['NCT'],
                    'model_output': response_text,
                    'extracted_nct': extracted_nct,
                    'correct': is_correct,
                    'urls': ', '.join(urls) if urls else ''
                }
                processed_results.append(result_dict)
                
            except Exception as e:
                logger.error(f"Error processing result {i}: {str(e)}")
                logger.error(traceback.format_exc())
                # Add a placeholder result with error information
                processed_results.append({
                    'evidence': row['evidence'] if 'evidence' in row else 'Unknown',
                    'correct_nct': row['NCT'] if 'NCT' in row else 'Unknown',
                    'model_output': f"ERROR: {str(e)}",
                    'extracted_nct': '',
                    'correct': False,
                    'urls': ''  # Empty URLs for error cases
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
    parser = argparse.ArgumentParser(description="Evaluate model's ability to extract NCT numbers")
    parser.add_argument("csv_path", help="Path to CSV file with evidence and NCT columns")
    parser.add_argument("--model", default="gemini-2.0-flash", choices=list(GEMINI_MODELS.keys()), 
                        help="Gemini model to use")
    parser.add_argument("--use_tools", action="store_true", help="Use Google Search tool")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--output", help="Output CSV path (optional)")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 8 examples")
    parser.add_argument("-n", type=int, help="Process only the first n rows (for testing)")
    parser.add_argument("--debug", action="store_true", help="Save debug information")
    
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
        n=args.n
    )

if __name__ == "__main__":
    main()