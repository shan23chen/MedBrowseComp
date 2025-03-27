#!/usr/bin/env python3
"""
Demo script for Gemini inference on HemOnc predictions
"""
import os
import argparse
from dotenv import load_dotenv
from gemini_inference import GeminiInference, GEMINI_MODELS
from process_predictions import process_csv

# Load environment variables from .env file
load_dotenv()

def check_environment():
    """Check if the environment is properly set up"""
    if not os.environ.get("GEMINI_API_KEY"):
        print("\033[91mWarning: GEMINI_API_KEY not found in environment variables\033[0m")
        print("Please create a .env file with your API key like this:")
        print("GEMINI_API_KEY=your_api_key_here")
        return False
    return True

def run_direct_inference_demo():
    """Run a demo of direct inference without using CSV data"""
    if not check_environment():
        return
    
    print("\n\033[1m=== Direct Inference Demo ===\033[0m")
    
    # Example evidence and question
    evidence = """Thalidomide maintenance treatment increases progression-free but not overall survival 
    in elderly patients with myeloma. Thalidomide maintenance therapy after stem cell transplantation 
    resulted in increased progression-free survival and overall survival in a few trials but its role 
    in non-transplant eligible patients with multiple myeloma remains unclear."""
    
    question = "Choose an option that best describes the efficacy of thalidomide maintenance treatment."
    options = [
        "Increases both progression-free and overall survival",
        "Increases progression-free but not overall survival",
        "No significant effect on progression-free or overall survival"
    ]
    
    print("\nTesting inference without tools:")
    for model_name in ["gemini-1.5-flash", "gemini-2.0-flash"]:
        try:
            inference = GeminiInference(model_name=model_name)
            
            # Format the prompt
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
            
            print(f"\nRunning inference with {model_name}...")
            response = inference.generate_response(prompt, use_tools=False)
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")

def run_csv_processing_demo(csv_path, max_examples=2):
    """Run a demo of CSV processing with a limited number of examples"""
    if not check_environment():
        return
        
    print("\n\033[1m=== CSV Processing Demo ===\033[0m")
    
    try:
        import pandas as pd
        
        # Read just a few examples from the CSV
        df = pd.read_csv(csv_path)
        sample_df = df.head(max_examples)
        
        # Save sample to a temporary file
        temp_csv = "temp_sample.csv"
        sample_df.to_csv(temp_csv, index=False)
        
        print(f"\nProcessing {max_examples} examples from {csv_path}")
        
        # Process with default model (no tools)
        results = process_csv(
            csv_path=temp_csv,
            model_name="gemini-2.0-flash",
            use_tools=False,
            max_workers=2
        )
        
        # Clean up
        os.remove(temp_csv)
        
    except Exception as e:
        print(f"Error in CSV demo: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Gemini Inference Demo")
    parser.add_argument("--csv", action="store_true", help="Run CSV processing demo")
    parser.add_argument("--inference", action="store_true", help="Run direct inference demo")
    parser.add_argument("--csv_path", default="data/wrong_predictions.csv", help="Path to CSV file")
    parser.add_argument("--max_examples", type=int, default=2, help="Maximum examples to process in demo")
    
    args = parser.parse_args()
    
    # If no specific demo is requested, run both
    if not args.csv and not args.inference:
        args.inference = True
        args.csv = True
    
    print("\033[1m=== Gemini Inference Demo ===\033[0m")
    print("Available models:", ", ".join(GEMINI_MODELS.keys()))
    
    if args.inference:
        run_direct_inference_demo()
    
    if args.csv:
        run_csv_processing_demo(args.csv_path, args.max_examples)
    
    print("\n\033[1m=== Demo Complete ===\033[0m")
    print("\nTo run inference on the full dataset:")
    print("python process_predictions.py --model gemini-2.0-flash --threads 4")
    print("\nTo use tools:")
    print("python process_predictions.py --model gemini-2.0-flash --use_tools")
    print("\nTo save results to a CSV:")
    print("python process_predictions.py --output results.csv")
    print("\nTo run in test mode (only 8 examples):")
    print("python process_predictions.py --test")

if __name__ == "__main__":
    main()
