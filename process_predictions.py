import csv
import os
import re
import json
import pandas as pd
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from gemini_inference_v2 import GeminiInference, run_inference_multithread, GEMINI_MODELS

def extract_answer(response: Optional[str]) -> str:
    """
    Extract the answer from model response in the format <answer>Option X</answer>
    
    Args:
        response: Model response text
        
    Returns:
        Extracted answer or full response if answer not found
    """
    # Add check for None or non-string input
    if not isinstance(response, str):
        return "Error: Invalid response type"
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

def run_inference_and_save_raw(
    csv_path: str,
    raw_output_path: str,
    model_name: str = "gemini-2.0-flash",
    use_tools: bool = False,
    max_workers: int = 4,
    test_mode: bool = False
):
    """
    Runs model inference on data from a CSV and saves raw results to a JSON file.

    Args:
        csv_path: Path to the input CSV file.
        raw_output_path: Path to save the intermediate raw results (JSON).
        model_name: Gemini model to use.
        use_tools: Whether to use Google Search tool.
        max_workers: Number of parallel threads for inference.
        test_mode: Whether to run in test mode (only 8 examples).
    """
    try:
        # Read CSV file
        print(f"Reading input CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        # Ensure 'answer' column exists and rename it to 'actual' for clarity
        if 'answer' not in df.columns:
            raise ValueError("Input CSV must contain an 'answer' column.")
        df['actual'] = df['answer']

        # If test mode, limit to 8 examples
        if test_mode:
            df = df.head(8)
            print("TEST MODE: Limited to 8 examples")
        
        # Create list of inputs and store original rows
        inputs = []
        original_rows_data = [] # Store row data as dicts for JSON serialization

        print("Preparing input data...")
        required_cols = ['evidence', 'question 1', 'option 1', 'option 2', 'option 3', 'actual']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(missing_cols)}")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing inputs"):
            evidence = row['evidence']
            question = row['question 1']
            options = [row['option 1'], row['option 2'], row['option 3']]

            # Format the prompt
            prompt = format_prompt(evidence, question, options)
            inputs.append(prompt)
            original_rows_data.append(row.to_dict()) # Convert row to dict
        
        # Run inference in parallel
        print(f"Processing {len(inputs)} examples using {model_name}" + 
              f" {'with' if use_tools else 'without'} tools" +
              f" using {max_workers} threads...")

        # Run inference in parallel
        print(f"Processing {len(inputs)} examples using {model_name}" +
              f" {'with' if use_tools else 'without'} tools" +
              f" using {max_workers} threads...")

        model_outputs = run_inference_multithread(
            model_name=model_name,
            input_list=inputs,
            use_tools=use_tools,
            max_workers=max_workers
        )

        # Combine original data with model outputs
        raw_results_data = []
        for original_row_dict, model_output in zip(original_rows_data, model_outputs):
            # Explicitly select and type-cast necessary fields for JSON storage
            data_to_save = {
                'evidence': str(original_row_dict.get('evidence', '')),
                'question 1': str(original_row_dict.get('question 1', '')),
                'option 1': str(original_row_dict.get('option 1', '')),
                'option 2': str(original_row_dict.get('option 2', '')),
                'option 3': str(original_row_dict.get('option 3', '')),
                # Ensure 'actual' is stored as an int. Handle potential missing/NaN values.
                'actual': int(original_row_dict['actual']) if pd.notna(original_row_dict.get('actual')) else None
            }
            # Ensure model output is stored as a string, even if it's None or an error object somehow slipped through
            safe_model_output = model_output if isinstance(model_output, str) else str(model_output)

            raw_results_data.append({
                'original_data': data_to_save,
                'model_output': safe_model_output
            })

        # Save raw results to JSON file
        print(f"Saving raw results to {raw_output_path}...")
        with open(raw_output_path, 'w') as f:
            json.dump(raw_results_data, f, indent=2)
        print(f"Raw results saved successfully.")

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {csv_path}")
    except ValueError as ve:
        print(f"Error preparing data: {str(ve)}")
    except Exception as e:
        print(f"An unexpected error occurred during inference: {str(e)}")


def process_raw_results(
    raw_input_path: str,
    output_path: str,
    include_evidence: bool = False
):
    """
    Processes raw results from a JSON file, extracts answers, calculates accuracy,
    and saves the final processed results to a CSV file.

    Args:
        raw_input_path: Path to the intermediate raw results JSON file.
        output_path: Path to save the final processed results CSV.
        include_evidence: Whether to include evidence text in the output CSV.
    """
    try:
        # Load raw results from JSON file
        print(f"Loading raw results from {raw_input_path}...")
        with open(raw_input_path, 'r') as f:
            raw_results_data = json.load(f)
        print(f"Loaded {len(raw_results_data)} raw results.")

        # Process results
        processed_results = []
        print("Processing results...")
        for item in tqdm(raw_results_data, desc="Processing results"):
            original_data = item['original_data']
            model_output = item['model_output']

            # Extract answer from response
            extracted_answer = extract_answer(model_output)

            # Create result dictionary
            input_dict = {
                'question': original_data.get('question 1', 'N/A'),
                'options': [
                    original_data.get('option 1', 'N/A'),
                    original_data.get('option 2', 'N/A'),
                    original_data.get('option 3', 'N/A')
                ]
            }

            # Include evidence if requested
            if include_evidence:
                input_dict['evidence'] = original_data.get('evidence', 'N/A')

            actual_answer_num = original_data.get('actual')
            actual_option = f"Option {actual_answer_num}" if actual_answer_num is not None else "N/A"

            result_dict = {
                'input': json.dumps(input_dict), # Store input dict as JSON string for CSV
                'model_output': model_output,
                'extracted_answer': extracted_answer,
                'actual_option': actual_option,
                'correct': extracted_answer == actual_option and actual_option != "N/A"
            }
            processed_results.append(result_dict)

        # Calculate accuracy
        correct_count = sum(1 for r in processed_results if r['correct'])
        total_valid = sum(1 for r in processed_results if r['actual_option'] != "N/A")
        accuracy = correct_count / total_valid if total_valid > 0 else 0
        print(f"Accuracy: {correct_count}/{total_valid} ({accuracy:.2%})")

        # Save results to CSV
        print(f"Saving processed results to {output_path}...")
        output_df = pd.DataFrame(processed_results)
        output_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Processed results saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Raw results file not found at {raw_input_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {raw_input_path}. File might be corrupted or empty.")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Run inference and process results for Hemonc benchmark using Gemini models.")
    parser.add_argument("--csv_path", default="data/Hemonc.csv", help="Path to the input CSV file (required for inference).")
    parser.add_argument("--model", default="gemini-2.0-flash", choices=list(GEMINI_MODELS.keys()),
                        help="Gemini model to use for inference.")
    parser.add_argument("--use_tools", action="store_true", help="Use Google Search tool during inference.")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for parallel inference.")
    parser.add_argument("--raw_output", default="raw_results.json", help="Path to save/load intermediate raw results (JSON).")
    parser.add_argument("--output", default="processed_results.csv", help="Path to save the final processed results (CSV).")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 8 examples during inference.")
    parser.add_argument("--include-evidence", action="store_true", help="Include evidence text in the final output CSV.")
    parser.add_argument("--run_inference", action=argparse.BooleanOptionalAction, default=True, help="Run the inference step.")
    parser.add_argument("--process_results", action=argparse.BooleanOptionalAction, default=True, help="Process the results step.")

    args = parser.parse_args()

    if args.model not in GEMINI_MODELS:
        print(f"Invalid model name: {args.model}")
        print(f"Available models: {', '.join(GEMINI_MODELS.keys())}")
        return

    if args.run_inference:
        print("--- Running Inference Step ---")
        run_inference_and_save_raw(
            csv_path=args.csv_path,
            raw_output_path=args.raw_output,
            model_name=args.model,
            use_tools=args.use_tools,
            max_workers=args.threads,
            test_mode=args.test
        )
        print("--- Inference Step Complete ---")
    else:
        print("Skipping inference step as per --no-run_inference flag.")

    if args.process_results:
        # Check if raw results file exists if inference was skipped
        if not args.run_inference and not os.path.exists(args.raw_output):
             print(f"Error: Cannot process results. Raw results file '{args.raw_output}' not found and inference step was skipped.")
             return

        if os.path.exists(args.raw_output):
            print("\n--- Processing Results Step ---")
            process_raw_results(
                raw_input_path=args.raw_output,
                output_path=args.output,
                include_evidence=args.include_evidence
            )
            print("--- Processing Results Step Complete ---")
        elif args.run_inference:
             # This case might happen if inference failed to produce the file
             print(f"Warning: Raw results file '{args.raw_output}' not found even after running inference. Skipping processing.")
        # else: # Handled above: not args.run_inference and not os.path.exists
            # print(f"Error: Cannot process results. Raw results file '{args.raw_output}' not found and inference step was skipped.")

    else:
        print("Skipping processing results step as per --no-process_results flag.")


if __name__ == "__main__":
    main()
