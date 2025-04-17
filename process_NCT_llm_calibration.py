import argparse
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import re
from openai_search_inference import OpenAISearchInference
from gemini_inference import GeminiInference, GEMINI_MODELS
from sonar_inference import SONAR_MODELS
from openai_search_inference import OPENAI_SEARCH_MODELS

def parse_llm_judge_output(judge_result):
    """
    Parse the LLM judge output for extracted_final_answer, correct, confidence, and reasoning.
    """
    extracted_final_answer = None
    llm_correct = None
    llm_confidence = None
    llm_reasoning = None
    
    # Extract fields
    answer_match = re.search(r'extracted_final_answer:\s*(.*)', judge_result)
    correct_match = re.search(r'correct:\s*(yes|no)', judge_result, re.IGNORECASE)
    conf_match = re.search(r'confidence:\s*(\d+)', judge_result)
    reasoning_match = re.search(r'reasoning:\s*(.*?)(?:\n|$)', judge_result, re.IGNORECASE)

    if answer_match:
        extracted_final_answer = answer_match.group(1).strip()
    if correct_match:
        llm_correct = correct_match.group(1).strip().lower()
    if conf_match:
        llm_confidence = int(conf_match.group(1))
    if reasoning_match:
        llm_reasoning = reasoning_match.group(1).strip()
    return extracted_final_answer, llm_correct, llm_confidence, llm_reasoning


def main():
    parser = argparse.ArgumentParser(description="Run LLM judge/calibration on model outputs CSV.")
    parser.add_argument("--input_csv", required=True, help="Input CSV (from process_NCT_predictions.py output)")
    parser.add_argument("--output_csv", required=True, help="Output CSV to save LLM judge results")
    parser.add_argument("--model", required=True, help="LLM model to use (Gemini or OpenAI)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("LLMCalibration")

    df = pd.read_csv(args.input_csv)
    logger.info(f"Loaded {len(df)} rows from {args.input_csv}")

    # Model routing
    model_name = args.model
    is_gemini = model_name in GEMINI_MODELS
    is_openai = model_name in OPENAI_SEARCH_MODELS
    is_sonar = model_name in SONAR_MODELS
    if is_sonar:
        raise ValueError("Sonar models are not supported for LLM judge/calibration.")
    if not (is_gemini or is_openai):
        raise ValueError("Model must be Gemini or OpenAI.")

    if is_gemini:
        judge_model = GeminiInference(model_name=model_name)
    elif is_openai:
        base_model = model_name.rstrip("-search-preview")
        judge_model = OpenAISearchInference(model_name=base_model)

    JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the\nprecise and unambiguous [correct_answer] below.\n[question]: {question}\n[response]: {response}\nYour judgement must be in the format and criteria specified below:\nextracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer\nas ’None’ if there is no exact, final answer to extract from the response.\n[correct_answer]: {correct_answer}\nreasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer],\nfocusing only on if there are meaningful differences between [correct_answer] and the\nextracted_final_answer. Do not comment on any background to the problem, do not attempt to solve\nthe problem, do not argue for any answer different than [correct_answer], focus only on whether the\nanswers match.\ncorrect: Answer ’yes’ if extracted_final_answer matches the [correct_answer] given above, or is within\na small margin of error for numerical problems. Answer ’no’ otherwise, i.e. if there if there is\nany inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.\nconfidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there\nis no confidence score available."""

    llm_rows = []
    calibration_pairs = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM Calibration Judging"):
        question = row.get('question', '')
        correct_answer = row.get('correct_answer', '')
        model_output = row.get('model_output', '')
        regex_correct = row.get('correct', '')
        prompt = JUDGE_PROMPT.format(question=question, response=model_output, correct_answer=correct_answer)
        try:
            judge_result = judge_model.call(prompt)
        except Exception as e:
            logger.error(f"LLM judge error on row {idx}: {e}")
            judge_result = ''
        extracted_final_answer, llm_correct, llm_confidence, llm_reasoning = parse_llm_judge_output(judge_result)
        llm_rows.append({
            'question': question,
            'correct_answer': correct_answer,
            'model_output': model_output,
            'llm_extracted_final_answer': extracted_final_answer,
            'llm_correct': llm_correct,
            'llm_confidence': llm_confidence,
            'llm_reasoning': llm_reasoning,
            'regex_correct': regex_correct
        })
        # For calibration error
        if llm_confidence is not None and llm_correct is not None:
            calibration_pairs.append((llm_confidence/100.0, 1 if llm_correct == 'yes' else 0))

    # Save to output CSV
    out_df = pd.DataFrame(llm_rows)
    out_df.to_csv(args.output_csv, index=False)
    logger.info(f"LLM judge results saved to {args.output_csv}")

    # Compute RMS calibration error
    if calibration_pairs:
        bins = np.arange(0, 1.01, 0.1)
        rmses = []
        for i in range(len(bins)-1):
            bin_lower, bin_upper = bins[i], bins[i+1]
            bin_pairs = [x for x in calibration_pairs if bin_lower <= x[0] < bin_upper]
            if bin_pairs:
                acc = np.mean([x[1] for x in bin_pairs])
                conf = np.mean([x[0] for x in bin_pairs])
                rmses.append((acc - conf) ** 2)
        rms_calib_error = np.sqrt(np.mean(rmses)) if rmses else float('nan')
        logger.info(f"LLM RMS Calibration Error: {rms_calib_error:.4f}")
    else:
        logger.warning("No calibration pairs collected for RMS calibration error.")

if __name__ == "__main__":
    main()
