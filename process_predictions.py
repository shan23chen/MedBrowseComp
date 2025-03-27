import csv
import os
import re
import pandas as pd
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm
from gemini_inference import GeminiInference, run_inference_multithread, GEMINI_MODELS

def extract_answer(response: str) -> str:
    """
    Extract the answer from model response in the format <answer>Option X</answer>
    
    Args:
        response: Model response text
        
    Returns:
        Extracted answer or full response if answer not found
    """
    match = re.search(r'<answer>(Option \d+)</answer>', response)
    if match:
        return match.group(1)
    return response

def format_prompt(evidence: str, question: str, options: List[str]) -> str:
    """
    Format the prompt for the model
    
    Args:
        evidence: The evidence text
        question: The question to answer
        options: List of options
        
    Returns:
        Formatted prompt
    """
    prompt = f"""Read the following medical evidence and select the most appropriate option that answers the question.

Evidence:
{evidence}

Question:
{question}

Options:
- Option 1: {options[0]}
- Option 2: {options[1]}
- Option 3: {options[2]}

Provide your answer in this exact format: <answer>Option X</answer>, where X is the number of the correct option (1, 2, or 3).
"""
    return prompt

def process_csv(
    csv_path: str,
    model_name: str = "gemini-2.0-flash",
    use_tools: bool = False,
    max_workers: int = 4,
    output_path: str = None,
    test_mode: bool = False
) -> List[Dict]:
    """
    Process CSV file with predictions
    
    Args:
        csv_path: Path to CSV file
        model_name: Gemini model to use
        use_tools: Whether to use Google Search tool
        max_workers: Number of parallel threads
        output_path: Optional path for saving results CSV
        test_mode: Whether to run in test mode (only 8 examples)
        
    Returns:
        List of result dictionaries
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        df['actual'] = df['answer']
        
        # If test mode, limit to 8 examples
        if test_mode:
            df = df.head(8)
            print("TEST MODE: Limited to 8 examples")
        
        # Create list of inputs
        inputs = []
        rows = []
        
        print("Preparing input data...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing inputs"):
            evidence = row['evidence']
            question = row['question 1']  # Using question 1 as specified
            options = [
                row['option 1'],
                row['option 2'],
                row['option 3']
            ]
            try:
                actual = row['actual']
            except KeyError:
                print("Warning: 'actual' column not found in CSV. Assuming first option as actual.")
                actual = options[0]
            
            # Format the prompt
            prompt = format_prompt(evidence, question, options)
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
        print("Processing results...")
        for result, row in tqdm(zip(results, rows), total=len(results), desc="Processing results"):
            # Extract answer from response
            extracted_answer = extract_answer(result)
            
            # Create result dictionary - excluding evidence from input to save space
            result_dict = {
                'input': {
                    'question': row['question 1'],
                    'options': [
                        row['option 1'],
                        row['option 2'],
                        row['option 3']
                    ]
                },
                'model_output': result,
                'extracted_answer': extracted_answer,
                'actual_option': f"Option {row['actual']}",
                'correct': extracted_answer == f"Option {row['actual']}"
            }
            processed_results.append(result_dict)
        
        # Calculate accuracy
        correct_count = sum(1 for r in processed_results if r['correct'])
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
    parser = argparse.ArgumentParser(description="Process wrong predictions using Gemini models")
    parser.add_argument("--csv_path", default="data/wrong_predictions.csv", help="Path to CSV file")
    parser.add_argument("--model", default="gemini-2.0-flash", choices=list(GEMINI_MODELS.keys()), 
                        help="Gemini model to use")
    parser.add_argument("--use_tools", action="store_true", help="Use Google Search tool")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel processing")
    parser.add_argument("--output", help="Output CSV path (optional)")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 8 examples")
    
    args = parser.parse_args()
    
    if args.model not in GEMINI_MODELS:
        print(f"Invalid model name: {args.model}")
        print(f"Available models: {', '.join(GEMINI_MODELS.keys())}")
        return
    
    print(f"Processing {args.csv_path} with {args.model}")
    process_csv(
        csv_path=args.csv_path,
        model_name=args.model,
        use_tools=args.use_tools,
        max_workers=args.threads,
        output_path=args.output,
        test_mode=args.test
    )

if __name__ == "__main__":
    main()
