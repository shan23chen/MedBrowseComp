import csv
import os
import re
import pandas as pd
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm
from gemini_inference_v2 import GeminiInference, run_inference_multithread, GEMINI_MODELS

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
    prompt = f"""Give me the clinical trial id that showed this, start with NCT: {evidence}
    
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
        df = pd.read_csv(csv_path)
        
        # If test mode, limit to 8 examples
        if test_mode:
            df = df.head(8)
            print("TEST MODE: Limited to 8 examples")
        elif n is not None:
            df = df.head(n)
            print(f"Processing first {n} examples")
        
        # Create list of inputs
        inputs = []
        rows = []
        
        print("Preparing input data...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing inputs"):
            evidence = row['evidence']
            correct_nct = row['NCT']
            
            # Format the prompt
            prompt = format_nct_prompt(evidence)
            inputs.append(prompt)
            rows.append(row)
        
        # Run inference in parallel
        print(f"Processing {len(inputs)} examples using {model_name}" + 
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
        print("Processing results...")
        for result, row in tqdm(zip(results, rows), total=len(results), desc="Processing results"):
            # Extract NCT from response
            extracted_nct = extract_nct_from_response(result)
            
            # Check if the extracted NCT is in the correct NCT (which might have multiple NCT numbers)
            is_correct = False
            nct_list = [nct.strip() for nct in row['NCT'].split(',')]
            if extracted_nct in nct_list:
                is_correct = True
                correct_count += 1
            
            # Create result dictionary
            result_dict = {
                'evidence': row['evidence'],
                'correct_nct': row['NCT'],
                'model_output': result,
                'extracted_nct': extracted_nct,
                'correct': is_correct
            }
            processed_results.append(result_dict)
        
        # Calculate accuracy
        accuracy = correct_count / len(processed_results) if processed_results else 0
        print(f"Accuracy: {correct_count}/{len(processed_results)} ({accuracy:.2%})")
        
        # Save results to CSV if output path provided
        if output_path:
            print(f"Saving results to {output_path}...")
            output_df = pd.DataFrame(processed_results)
            output_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return processed_results
        
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
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
    
    args = parser.parse_args()
    
    if args.model not in GEMINI_MODELS:
        print(f"Invalid model name: {args.model}")
        print(f"Available models: {', '.join(GEMINI_MODELS.keys())}")
        return
    
    print(f"Processing {args.csv_path} with {args.model}")
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