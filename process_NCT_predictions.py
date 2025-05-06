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
from pydantic import BaseModel, Field
from openai import OpenAI
from gemini_inference import GeminiInference, run_inference_multithread as gemini_run_inference_multithread, GEMINI_MODELS
from sonar_inference import SonarInference, run_inference_multithread as sonar_run_inference_multithread, SONAR_MODELS
from openai_search_inference import OpenAISearchInference, run_inference_multithread as openai_search_run_inference_multithread, OPENAI_SEARCH_MODELS

ALL_MODELS = {}
ALL_MODELS.update(GEMINI_MODELS)
ALL_MODELS.update(SONAR_MODELS)
ALL_MODELS.update(OPENAI_SEARCH_MODELS)

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

# --- LLM as Judge Setup ---
client = OpenAI() # Assumes OPENAI_API_KEY is set in environment

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the
precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: {response}
Your judgement must be in the format and criteria specified below:
extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer
as 'None' if there is no exact, final answer to extract from the response.
[correct_answer]: {correct_answer}
reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer],
focusing only on if there are meaningful differences between [correct_answer] and the
extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve
the problem, do not argue for any answer different than [correct_answer], focus only on whether the
answers match.
correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within
a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is
any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
"""

class JudgeOutput(BaseModel):
    extracted_final_answer: str = Field(description="The final exact answer extracted from the [response]. Put 'None' if no exact answer found.")
    reasoning: str = Field(description="Explanation of correctness based ONLY on comparing extracted_final_answer and correct_answer.")
    correct: str = Field(description="Must be 'yes' or 'no'.")

# --- End LLM as Judge Setup ---

def judge_response(question: str, response: str, correct_answer: str, judge_model: str = "gpt-4.1-mini") -> JudgeOutput:
    """
    Uses an LLM to judge the correctness of a response against a correct answer.

    Args:
        question: The original question asked.
        response: The model's response to the question.
        correct_answer: The ground truth answer.
        judge_model: The OpenAI model to use for judging.

    Returns:
        A JudgeOutput object containing the evaluation.
    """
    judge_input_text = JUDGE_PROMPT.format(
        question=question,
        response=response,
        correct_answer=correct_answer
    )
    try:
        judge_api_response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are an impartial judge evaluating an AI response based on provided criteria. Respond ONLY with a valid JSON object matching the requested structure."},
                {"role": "user", "content": judge_input_text}
            ],
            response_format={"type": "json_object"} # Ensure the output is a JSON object
        )
        # The model should return a JSON string in the message content based on the prompt instructions
        # Need to handle potential JSON parsing errors here as well
        try:
            parsed_output = JudgeOutput.model_validate_json(judge_api_response.choices[0].message.content)
        except json.JSONDecodeError as json_e:
             logger.error(f"Judge LLM did not return valid JSON: {json_e}")
             logger.error(f"Raw judge response: {judge_api_response.choices[0].message.content}")
             return JudgeOutput(
                 extracted_final_answer="JUDGE_JSON_ERROR",
                 reasoning=f"Judge LLM failed to produce valid JSON: {json_e}",
                 correct="no"
             )
        return parsed_output
    except Exception as e:
        logger.error(f"Error calling Judge LLM: {e}")
        logger.error(traceback.format_exc())
        # Return a default error object
        return JudgeOutput(
            extracted_final_answer="JUDGE_ERROR",
            reasoning=f"Error during judging: {e}",
            correct="no" # Treat errors as incorrect
        )


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

    # --- Consolidated handler for filled50/filled121 FIRST ---
    # Ensure this block is executed ONLY for these tasks and returns directly.
    if task == "filled50" or task == "filled121":
        
        # 1. Ingredient (Keep existing patterns, seem to work)
        match = re.search(r'(?i)INGREDIENT:\s*([A-Z0-9\s\-]+)', response)
        if match: return match.group(1).strip()
        match = re.search(r'(?i)ingredient(?:\s+is|\s+name)?:\s*([A-Z0-9\s\-]+)', response)
        if match: return match.group(1).strip()
        match = re.search(r'(?<!\w)([A-Z]{3,}(?:\s+[A-Z]+)*(?:\s+HYDROCHLORIDE)?)', response)
        if match: return match.group(1).strip()

        # 2. Company (More robust pattern)
        # Look for COMPANY: Name first
        match = re.search(r'(?i)COMPANY:\s*([A-Za-z0-9\s\.,&\-]+(?: Inc\.?| LLC\.?| Corp\.?| Ltd\.?| CV)?)', response)
        if match: return match.group(1).strip()
        # Look for company name after "company name is" or similar
        match = re.search(r'(?i)company(?:\s+name)?\s*(?:is)?:?\s*([A-Za-z0-9\s\.,&\-]+(?: Inc\.?| LLC\.?| Corp\.?| Ltd\.?| CV)?)', response)
        if match: return match.group(1).strip()
        # Look for likely company names (e.g., capitalized words, possibly with suffixes)
        match = re.search(r'(?<!\w)([A-Z][A-Za-z0-9\s\.,&\-]+(?: Inc\.?| LLC\.?| Corp\.?| Ltd\.?| CV)?)(?!\w)', response)
        # Avoid matching simple ALL CAPS words unless they look like company names (e.g., contain Inc, LLC etc.)
        if match and (re.search(r'(?i) Inc\.?| LLC\.?| Corp\.?| Ltd\.?| CV', match.group(1)) or ' ' in match.group(1).strip()):
             return match.group(1).strip()

        # 3. Patent Expiration Year (YYYY) (More robust pattern)
        # Look for DATE: YYYY or similar first
        match = re.search(r'(?i)(?:patent\s+expir\w*|DATE)[:\s]*(\d{4})', response)
        if match: return match.group(1).strip()
        # Look for YYYY after mention of patent expiration
        match = re.search(r'(?i)patent\s+expir(?:ation|es?|y)(?:.*?)(?<!\d)(\d{4})(?!\d)', response)
        if match: return match.group(1).strip()
        # Look for Month DD, YYYY and extract year
        match = re.search(r'(?i)(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+(\d{4})', response)
        if match: return match.group(1).strip()
        # Look for just YYYY if other patterns fail (less specific)
        match = re.search(r'(?<!\d)(\d{4})(?!\d)', response)
        if match: return match.group(1).strip()


        # 4. Exclusivity Date (MM-DD-YYYY or NA) (More robust pattern)
        # Check for NA variations first
        if re.search(r'(?i)\b(?:N/?A|not applicable|not available|no\s+exclusivity(?:\s+date)?|unknown)\b', response): return "NA"
        # Look for DATE: MM-DD-YYYY or similar
        match = re.search(r'(?i)(?:exclusivity|DATE)[:\s]*(\d{2}-\d{2}-\d{4})', response)
        if match: return match.group(1).strip()
        # Look for M/D/YYYY format and convert
        match = re.search(r'(?i)(?:exclusivity|DATE)[:\s]*(\d{1,2})[/-](\d{1,2})[/-](\d{4})', response)
        if match:
            month, day, year = match.group(1).zfill(2), match.group(2).zfill(2), match.group(3)
            return f"{month}-{day}-{year}"
        # Look for Month DD, YYYY format and convert
        match = re.search(r'(?i)(?:exclusivity\s+date\s+is)?\s*(?:([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4}))', response)
        if match:
            month_map = {'january': '01', 'jan': '01', 'february': '02', 'feb': '02', 'march': '03', 'mar': '03', 'april': '04', 'apr': '04', 'may': '05', 'june': '06', 'jun': '06', 'july': '07', 'jul': '07', 'august': '08', 'aug': '08', 'september': '09', 'sep': '09', 'october': '10', 'oct': '10', 'november': '11', 'nov': '11', 'december': '12', 'dec': '12'}
            month_name = match.group(1).lower()
            month = month_map.get(month_name, '01') # Default to 01 if month name invalid
            day = match.group(2).zfill(2)
            year = match.group(3)
            return f"{month}-{day}-{year}"
        # Check again for NA if no date format matched
        if re.search(r'(?i)\b(?:N/?A|not applicable|not available|no\s+exclusivity(?:\s+date)?|unknown)\b', response): return "NA"


        # 5. Stock Info (TICKER: $PRICE or NOT LISTED) (More robust pattern)
        # Check for NOT LISTED variations first
        if re.search(r'(?i)\bNOT\s+LISTED\b', response): return "NOT LISTED"
        # Look for TICKER: $PRICE format
        match = re.search(r'(?i)((?:[A-Z]{1,5}|\$[A-Z]{1,4}))\s*[:\s]+\$?(\d+\.\d+)', response)
        if match:
            ticker = match.group(1).replace('$', '').strip()
            try: price = f"${float(match.group(2).replace('$', '')):.2f}"
            except ValueError: price = match.group(2) # Keep original if conversion fails
            return f"{ticker}: {price}"
        # Look for ticker and price mentioned separately
        ticker_match = re.search(r'(?i)(?:ticker|symbol)\s*:?\s*([A-Z]{1,5})\b', response)
        price_match = re.search(r'(?i)(?:price|opening)\s*:?\s*\$?(\d+\.\d+)', response)
        if ticker_match and price_match:
            ticker = ticker_match.group(1).strip()
            try: price = f"${float(price_match.group(1)):.2f}"
            except ValueError: price = price_match.group(1) # Keep original if conversion fails
            return f"{ticker}: {price}"
        # Try to extract just a float value if other patterns fail (less specific)
        price_match = re.search(r'\$?(\d+\.\d+)', response)
        if price_match:
            try:
                # Check if it looks like a price (e.g., not part of a date or version number)
                context = response[max(0, price_match.start()-10):min(len(response), price_match.end()+10)]
                if not re.search(r'\d[/-]\d', context): # Avoid matching dates like 12/31/2024
                     # Check if a ticker symbol is nearby
                    ticker_nearby = re.search(r'\b([A-Z]{1,5})\b', response[max(0, price_match.start()-30):price_match.start()])
                    if ticker_nearby:
                         ticker = ticker_nearby.group(1)
                         price_val = f"${float(price_match.group(1)):.2f}"
                         return f"{ticker}: {price_val}"
                    else:
                         # Return just the price if no ticker found nearby
                         return f"${float(price_match.group(1)):.2f}"
            except ValueError: pass # Ignore if conversion fails
        # Check again for NOT LISTED if no stock format matched
        if re.search(r'(?i)\bNOT\s+LISTED\b', response): return "NOT LISTED"

        # If none of the specific filled50/121 patterns matched, return empty string
        logger.warning(f"No specific pattern matched for filled50/121 task. Response: {response[:200]}...")
        return "" 

    # --- Other Task Handlers ---
    # IMPORTANT: These should ONLY run if task is NOT filled50/121
    elif task == "track_trial_ids":
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
        
    # --- Specific Task Handlers ---
    elif task == "regime_drug_class" or task == "Ingredient":
        # Look for INGREDIENT: format first
        match = re.search(r'(?i)INGREDIENT:\s*([A-Z0-9\s\-]+)', response)
        if match: return match.group(1).strip()
        # Alternate formats
        match = re.search(r'(?i)ingredient(?:\s+is|\s+name)?:\s*([A-Z0-9\s\-]+)', response)
        if match: return match.group(1).strip()
        # Search for all-caps words
        match = re.search(r'(?<!\w)([A-Z]{3,}(?:\s+[A-Z]+)*(?:\s+HYDROCHLORIDE)?)', response)
        if match: return match.group(1).strip()
        return ""
        
    elif task == "latest_company_approval" or task == "Applicant_Full_Name":
        # Look for COMPANY: format first
        match = re.search(r'(?i)COMPANY:\s*([A-Z0-9\s\-]+)', response)
        if match: return match.group(1).strip()
        # Alternative formats
        match = re.search(r'(?i)company(?:\s+name)?:\s*([A-Z0-9\s\-]+)', response)
        if match: return match.group(1).strip()
        # Look for company names in all caps
        match = re.search(r'(?<!\w)([A-Z][A-Z\s]+(?:\s+[A-Z]+)*(?:\s+LLC|\s+SUB|\s+CV)?)', response)
        if match: return match.group(1).strip()
        return ""
        
    elif task == "Patent_Expire_Date_Text":
        # Match date: YYYY format
        match = re.search(r'(?i)DATE:?\s*(\d{4})', response)
        if match: return match.group(1).strip()
        # Match just the year after mention of patent expiration
        match = re.search(r'(?i)patent\s+expir(?:ation|es?|y)(?:\s+date)?(?:\s+is)?(?:\s+on)?:?\s*(?:[A-Za-z]+\s+\d{1,2},?\s+)?(\d{4})', response)
        if match: return match.group(1).strip()
        # Try to extract a date in the format Month DD, YYYY -> return only year
        match = re.search(r'(?i)(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+(\d{4})', response)
        if match: return match.group(1).strip()
        return ""
        
    elif task == "Exclusivity_Date":
        # Check for NA, N/A first
        if re.search(r'(?i)(?:N/?A|no\s+date|not\s+available|no\s+exclusivity\s+date)', response): return "NA"
        # Look for MM-DD-YYYY format
        match = re.search(r'(?i)DATE:?\s*(\d{2}-\d{2}-\d{4})', response)
        if match: return match.group(1).strip()
        # Look for M/D/YYYY format and convert to MM-DD-YYYY
        match = re.search(r'(?i)DATE:?\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})', response)
        if match:
            month, day, year = match.group(1).zfill(2), match.group(2).zfill(2), match.group(3)
            return f"{month}-{day}-{year}"
        # Look for Month DD, YYYY format and convert
        match = re.search(r'(?i)(?:exclusivity|date)(?:\s+date)?:?\s*(?:([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4}))', response)
        if match:
            month_map = {'january': '01', 'jan': '01', 'february': '02', 'feb': '02', 'march': '03', 'mar': '03', 'april': '04', 'apr': '04', 'may': '05', 'june': '06', 'jun': '06', 'july': '07', 'jul': '07', 'august': '08', 'aug': '08', 'september': '09', 'sep': '09', 'october': '10', 'oct': '10', 'november': '11', 'nov': '11', 'december': '12', 'dec': '12'}
            month_name = match.group(1).lower()
            month = month_map.get(month_name, '01')
            day = match.group(2).zfill(2)
            year = match.group(3)
            return f"{month}-{day}-{year}"
        return "NA" # Default to NA if no date found
        
    elif task == "Open_on_Approval":
        # Check if not listed first
        if re.search(r'(?i)NOT\s+LISTED', response): return "NOT LISTED"
        # Look for stock ticker format: TICKER: $XX.XX or similar
        match = re.search(r'(?i)((?:[A-Z]{1,5}|\$[A-Z]{1,4}))\s*[:\.]\s*\$?(\d+\.\d+)', response)
        if match:
            ticker = match.group(1).replace('$', '')
            try: price = f"${float(match.group(2).replace('$', '')):.2f}"
            except ValueError: price = match.group(2)
            return f"{ticker}: {price}"
        # Alternative format: "ticker symbol: XXX" and "opening price: $XX.XX"
        ticker_match = re.search(r'(?i)(?:ticker|symbol|stock)(?:\s+symbol)?(?:\s+is)?:?\s*([A-Z]{1,5})', response)
        price_match = re.search(r'(?i)(?:opening|stock|share)(?:\s+price)?(?:\s+was)?(?:\s+is)?:?\s*\$?(\d+\.\d+)', response)
        if ticker_match and price_match:
            ticker = ticker_match.group(1)
            try: price = f"${float(price_match.group(1)):.2f}"
            except ValueError: price = price_match.group(1)
            return f"{ticker}: {price}"
        # Try to extract just a float value for stock price
        price_match = re.search(r'(?i)(\d+\.\d+)', response)
        if price_match:
            try: return float(price_match.group(1))
            except ValueError: pass
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
    
    elif task == "track_start_date" or task == "patent_expiration_date" or task == "exclusivity_Date":
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
    elif task == "track_drug_route":
        # Look for exact "Drug route: X" format first
        direct_pattern = r'(?i)drug\s*route:?\s*([a-z0-9\s]+)'
        match = re.search(direct_pattern, response)
        if match:
            return match.group(1).strip()
        
        # Alternative patterns
        route_patterns = [
            r'(?i)(?:administration|administered|given)(?:\s+via|\s+by|\s+through)?:?\s*([a-z0-9\s]+\b)',
            r'(?i)route\s*(?:of|for)?\s*(?:administration|delivery)?:?\s*([a-z0-9\s]+\b)',
            r'(?i)route:?\s*([a-z0-9\s]+\b)'
        ]
        
        for pattern in route_patterns:
            match = re.search(pattern, response)
            if match:
                route = match.group(1).strip().lower()
                # Filter for common route terms
                common_routes = ["oral", "intravenous", "iv", "subcutaneous", "topical", 
                                "intramuscular", "im", "inhaled", "intranasal", "unknown"]
                
                # Check if any common route is in the extracted text
                for common_route in common_routes:
                    if common_route in route:
                        return common_route.capitalize()
                
                # If no common route found, return the extracted text
                return route.capitalize()
        
        logger.warning("Failed to extract drug route.")
        return ""
        
    elif task == "track_drug_class":
        # Look for exact "Drug class: X" format first
        direct_pattern = r'(?i)drug\s*class(?:es)?:?\s*([a-z0-9\s\-]+)'
        match = re.search(direct_pattern, response)
        if match:
            return match.group(1).strip()
        
        # Alternative patterns
        class_patterns = [
            r'(?i)class(?:es)?\s*(?:of|for)?\s*(?:medication|drug|therapy|treatment)?:?\s*([a-z0-9\s\-]+\b)',
            r'(?i)(?:medication|drug)\s*(?:is\s*a|belongs\s*to)\s*([a-z0-9\s\-]+\b)',
            r'(?i)class:?\s*([a-z0-9\s\-]+\b)'
        ]
        
        for pattern in class_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
        
        logger.warning("Failed to extract drug class.")
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
    task: str = None,
    run_inference=None,
    inference_kwargs=None,
    use_judge: bool = False,
    judge_model: str = "gpt-4.1-mini"
) -> List[Dict]:
    """
    Process CSV file with NCT predictions, optionally using an LLM judge.
    
    Args:
        csv_path: Path to CSV file
        model_name: Gemini model to use
        use_tools: Whether to use Google Search tool
        max_workers: Number of parallel threads
        output_path: Optional path for saving results CSV
        test_mode: Whether to run in test mode (only 8 examples)
        n: Number of rows to process (for testing)
        task: Task to perform
        run_inference: The inference function to use.
        inference_kwargs: Arguments for the inference function.
        use_judge: Whether to use the LLM judge for evaluation.
        judge_model: The model name to use for the LLM judge.
        
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
            
            ### for computer use agent fill in the prompt
            if task == "track_trial_ids":
                # track trial ids
                question = "Find/search the clinical trial id" + row['question 1'].split('Choose an option')[1] + '\nOutput it in the format NCT<Number>'
                correct_answer = row['NCT']
            elif task == "track_second_authors":
                # track second authors (old format)
                question = "Find/search the second author of the paper" + row['question 1'].split('Choose an option')[1] + '\nOutput it in the format SA<Second Author>'
                correct_answer = row['authors'].split('|')[1] if '|' in row['authors'] else ""
            elif task == "track_pmids":
                # track pmids
                question = "Find/search the pubmed id of the paper" + row['question 1'].split('Choose an option')[1] + '\nOutput it in the format pmid<pubmed id>'
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
            elif task == "track_drug_route":
                # Track drug route
                question = "Find/search the route of administration for the drug used in the clinical trial " + row['question 1'].split('Choose an option')[1] + '\n\nIMPORTANT: Respond with ONLY the format "Drug route: X" where X is the route (oral, intravenous, etc). If unknown, respond with "Drug route: Unknown". Do not include any other text in your response.'
                correct_answer = row['drug_routes']
            elif task == "track_drug_class":
                # Track drug class
                question = "Find/search the drug class used in the clinical trial " + row['question 1'].split('Choose an option')[1] + '\n\nIMPORTANT: Respond with ONLY the format "Drug class: X" where X is the class (antibiotics, chemotherapy, etc). Do not include any other text in your response.'
                correct_answer = row['drug_classes']
            elif task == "track_second_authors_multiple_pmids":
                # Format: model needs to determine both PMID and second author
                question = "please only return the US stock ticker (or OTC ticker if applicable) of " + row['Applicant_Full_Name'] + " in all CAPS in the format of STOCK_TICKER: ticker else return NOT_LISTED" 
                correct_answer = row['Applicant']
            
            #### final 5 questions for deepresearch
            elif task == "Ingredient":
                if row['Ingredient'] == 'APALUTAMIDE':
                    question = "For clinical trial " + row['NCT'] + ". Among the " + row['effecacy_group'] + 'effective regimen ingredients; find which is the ingredient with the first letter start with ' + 'AP' + "\n return only the name of the ingredient or unknown in all CAPS in the format of INGREDIENT: name"
                question = "For clinical trial " + row['NCT'] + ". Among the " + row['effecacy_group'] + 'effective regimen ingredients; find which is the ingredient with the first letter start with ' + row['Ingredient'][0]  + "\n return only the name of the ingredient or unknown in all CAPS in the format of INGREDIENT: name"
                correct_answer = row['Ingredient']
            elif task == "Applicant_Full_Name":
                if row['Ingredient'] == 'APALUTAMIDE':
                    question = "For clinical trial " + row['NCT'] + ". Among the " + row['effecacy_group'] + 'effective regimen ingredients; find which is the ingredient with the first letter start with ' + 'AP' + ".\n\nThen, find which company has the latest FDA approval date up till Dec, 2024 for this identified ingredient. Note that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals. \nReturn only the company name in format of COMPANY: name."
                question = "First, for clinical trial " + row['NCT'] + " among the " + row['effecacy_group'] + " effective regimen ingredients, identify which ingredient starts with the letter " + row['Ingredient'][0] + ".\n\nThen, find which company has the latest FDA approval date up till Dec, 2024 for this identified ingredient. Note that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals. \nReturn only the company name in format of COMPANY: name."
                correct_answer = row['Applicant_Full_Name']
            elif task == "Patent_Expire_Date_Text":
                if row['Ingredient'] == 'APALUTAMIDE':
                    question = "For clinical trial " + row['NCT'] + ". Among the " + row['effecacy_group'] + 'effective regimen ingredients; find which is the ingredient with the first letter start with ' + 'AP' + ".\n\nThen, for this identified ingredient that was last approved up till Dec, 2024, when is its patent expiration date?\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\nReturn only the date only the year YYYY format of date: YYYY."
                question = "First, for clinical trial " + row['NCT'] + " among the " + row['effecacy_group'] + " effective regimen ingredients, identify which ingredient starts with the letter " + row['Ingredient'][0] + ".\n\nThen, for this identified ingredient that was last approved up till Dec, 2024, when is its patent expiration date?\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\nReturn only the date only the year YYYY format of date: YYYY."
                correct_answer = row['Patent_Expire_Date_Text']
            elif task == "Exclusivity_Date":
                if pd.isna(row['Exclusivity_Date']) or row['Exclusivity_Date'] is None or str(row['Exclusivity_Date']).strip() == '':
                    question = "For clinical trial " + row['NCT'] + " among the " + row['effecacy_group'] + " effective regimen ingredients, identify which ingredient starts with the letter " + row['Ingredient'][0] + ".\n\nThen, for this identified ingredient that was last approved up till Dec, 2024, when is its exclusivity date according to the FDA?\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\nIf no exclusivity date exists or if it's not available, respond with 'DATE: NA'. Otherwise, return only the date in MM-DD-YYYY format of date: MM-DD-YYYY."
                    correct_answer = "NA"
                else:
                    question = "For clinical trial " + row['NCT'] + " among the " + row['effecacy_group'] + " effective regimen ingredients, identify which ingredient starts with the letter " + row['Ingredient'][0] + ".\n\nThen, for this identified ingredient that was last approved up till Dec, 2024, when is its exclusivity date according to the FDA?\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\nIf no exclusivity date exists or if it's not available, respond with 'DATE: NA'. Otherwise, return only the date in MM-DD-YYYY format of date: MM-DD-YYYY."
                    correct_answer = row['Exclusivity_Date']
            elif task == "Open_on_Approval":
                if row['Ingredient'] == 'APALUTAMIDE':
                    question = "For clinical trial " + row['NCT'] + ". Among the " + row['effecacy_group'] + 'effective regimen ingredients; find which is the ingredient with the first letter start with ' + 'AP' + ".\n\nThen, determine which pharmaceutical company received the most recent FDA approval (up until December 2024) for this identified ingredient.\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\n\nIf this company is listed on any US stock market (including over-the-counter markets), provide:\n1. The stock ticker symbol\n2. The opening stock price on the FDA approval date\n\nIf the company is not listed on any US stock market, please indicate \"NOT LISTED\"."
                question = "For clinical trial " + row['NCT'] + ", review the " + row['effecacy_group'] + " effective regimen ingredients and identify which ingredient starts with the letter " + row['Ingredient'][0] + ".\n\nThen, determine which pharmaceutical company received the most recent FDA approval (up until December 2024) for this identified ingredient.\nNote that we are only look for overall FDA drug approval, not new indidcation, not supplemental approvals.\n\nIf this company is listed on any US stock market (including over-the-counter markets), provide:\n1. The stock ticker symbol\n2. The opening stock price on the FDA approval date\n\nIf the company is not listed on any US stock market, please indicate \"NOT LISTED\"."
                correct_answer = row['Applicant'] if 'Applicant' in row else ""
            
            ### for filled prompt already
            elif task =="filled50" or task == "filled121":
                question = row['prompt']
                correct_answer = row['gold']
                

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

        if run_inference is None:
            # Fallback to Gemini for backward compatibility
            from gemini_inference import run_inference_multithread as gemini_run_inference_multithread
            run_inference = gemini_run_inference_multithread
            inference_kwargs = {"model_name": model_name}
        if inference_kwargs is None:
            inference_kwargs = {"model_name": model_name}
        inference_kwargs = dict(inference_kwargs)
        # Add use_tools/max_workers if supported by the backend
        if "use_tools" in run_inference.__code__.co_varnames:
            inference_kwargs["use_tools"] = use_tools
        if "max_workers" in run_inference.__code__.co_varnames:
            inference_kwargs["max_workers"] = max_workers
        results = run_inference(
            input_list=inputs,
            **inference_kwargs
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
                judge_reasoning = "N/A (Judge not used)"
                judge_extracted_answer = "N/A (Judge not used)"

                if use_judge:
                    logger.debug(f"Using LLM Judge ({judge_model}) for row {i}...")
                    judge_output = judge_response(
                        question=prompt,
                        response=response_text, # Use the raw response text for the judge
                        correct_answer=str(answer), # Ensure answer is string
                        judge_model=judge_model
                    )
                    is_correct = judge_output.correct.lower() == 'yes'
                    judge_reasoning = judge_output.reasoning
                    judge_extracted_answer = judge_output.extracted_final_answer
                    # Use the judge's extracted answer if available and not 'None' or error
                    if judge_extracted_answer not in ["None", "JUDGE_ERROR"]:
                         extracted_info = judge_extracted_answer
                    else:
                         # Fallback to regex extraction if judge fails or extracts None
                         extracted_info = extract_from_response(response_text, task=task)
                    logger.debug(f"Judge decision: {'Correct' if is_correct else 'Incorrect'}. Reason: {judge_reasoning}")

                else:
                    # --- Original Rule-Based Correctness Check ---
                    extracted_info = extract_from_response(response_text, task=task) # Extract using regex

                    # Special case for "NOT LISTED" responses
                    if task == "Open_on_Approval" or task == "filled50" or task == "filled121":
                        if str(answer).strip() == "NOT LISTED" and extracted_info.strip() == "NOT LISTED":
                            is_correct = True
                    
                    if task == "track_trial_ids":
                        nct_list = [nct.strip() for nct in row['NCT'].split(',')]
                        is_correct = extracted_info in nct_list
                    elif task == "track_second_authors":
                        # Old format - direct comparison
                        is_correct = extracted_info.strip().lower() == answer.strip().lower()
                    elif task == "track_pmids":
                        # pmid_list = [str(int(pmid)) for pmid in row['pmids'].split(',')]
                        # is_correct = extracted_info in pmid_list
                        is_correct = str(extracted_info).strip() == str(answer).strip()
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
                    elif task == "track_start_date" or task == "patent_expiration_date" or task == "exclusivity_Date":
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
                    elif task == "track_drug_route":
                        # For drug route, normalize to handle variations
                        extracted_route = extracted_info.strip().lower()
                        expected_route = answer.strip().lower()
                        
                        # Direct match first
                        if extracted_route == expected_route:
                            is_correct = True
                        else:
                            # Check for variations and common abbreviations
                            route_mapping = {
                                "intravenous": ["iv", "i.v.", "i.v", "intra-venous"],
                                "intramuscular": ["im", "i.m.", "i.m", "intra-muscular"],
                                "subcutaneous": ["sc", "s.c.", "s.c", "sub-cutaneous", "subcut"],
                                "oral": ["by mouth", "p.o.", "po", "per os"],
                                "unknown": ["not specified", "not reported", "not stated", "unclear"]
                            }
                            
                            # Check if expected route has any known variations
                            for main_route, variations in route_mapping.items():
                                if expected_route == main_route:
                                    # If expected is a main route, check if extracted is a variation
                                    is_correct = extracted_route in variations
                                    break
                                elif expected_route in variations:
                                    # If expected is a variation, check if extracted is the main route or another variation
                                    is_correct = extracted_route == main_route or extracted_route in variations
                                    break
                            else:
                                # If no match found in mappings, fall back to direct comparison
                                is_correct = False

                    elif task == "track_drug_class":
                        # For drug class, use basic case-insensitive comparison
                        extracted_class = extracted_info.strip().lower()
                        expected_class = answer.strip().lower()
                        
                        # Direct match first
                        if extracted_class == expected_class:
                            is_correct = True
                        else:
                            # Check for partial matches for compound drug classes
                            expected_classes = [cls.strip() for cls in expected_class.split('|')]
                            
                            # Check if the extracted class contains any of the expected classes
                            is_correct = any(exp_cls in extracted_class for exp_cls in expected_classes)
                            
                            # Also check if expected class contains the extracted class
                            if not is_correct:
                                is_correct = any(extracted_class in exp_cls for exp_cls in expected_classes)
                    elif task == "regime_drug_class" or task == "latest_company_approval" or task == "filled50" or task == "filled121":
                        # For filled50/filled121 tasks, try direct string comparison first
                        if task == "filled50" or task == "filled121":
                            # Try direct case-sensitive match first (many gold answers are in ALL CAPS)
                            if extracted_info.strip() == answer.strip():
                                is_correct = True
                            else:
                                # Try case-insensitive match if direct match fails
                                extracted_upper = extracted_info.strip().upper()
                                expected_upper = answer.strip().upper()
                                
                                if extracted_upper == expected_upper:
                                    is_correct = True
                                elif answer.strip() == "NOT LISTED" and "NOT LISTED" in extracted_info.upper():
                                    is_correct = True
                                # For numeric values (like stock prices)
                                elif re.match(r'^\d+\.\d+$', answer.strip()) and re.match(r'^\d+\.\d+$', extracted_info.strip()):
                                    try:
                                        # Allow small difference for floating point values
                                        expected_float = float(answer.strip()) 
                                        extracted_float = float(extracted_info.strip())
                                        # Allow 1% tolerance for stock prices
                                        tolerance = expected_float * 0.01
                                        is_correct = abs(expected_float - extracted_float) <= tolerance
                                    except ValueError:
                                        is_correct = False
                        else:
                            # For other tasks, use the existing logic
                            extracted_class = extracted_info.strip().lower()
                            expected_class = answer.strip().lower()
                            
                            # Direct match first
                            if extracted_class == "":
                                is_correct = False
                            else: 
                                if extracted_class == expected_class:
                                    is_correct = True
                                else:
                                    # Check for partial matches for compound drug classes
                                    expected_classes = [cls.strip() for cls in expected_class.split(' ')]
                                    
                                    # Check if the extracted class contains any of the expected classes
                                    is_correct = any(exp_cls in extracted_class for exp_cls in expected_classes)
                                    
                                    # Also check if expected class contains the extracted class
                                    if not is_correct:
                                        is_correct = any(extracted_class in exp_cls for exp_cls in expected_classes)
                    # --- End Original Rule-Based Correctness Check ---

                if is_correct:
                    correct_count += 1
                
                # Create result dictionary
                result_dict = {
                    'question': prompt,
                    'correct_answer': answer,
                    'model_output': response_text,
                    'extracted_info': extracted_info, # This might be from regex or judge
                    'correct': is_correct,
                    'urls': ', '.join(urls) if urls else '',
                    'judge_reasoning': judge_reasoning,
                    'judge_extracted_answer': judge_extracted_answer
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
                    'urls': '',
                    'judge_reasoning': f"ERROR: {str(e)}",
                    'judge_extracted_answer': "ERROR"
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
    parser.add_argument("--model", default="gemini-2.0-flash", choices=list(ALL_MODELS.keys()), 
                        help="Model to use (Gemini, Sonar, or OpenAI Search)")
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
                                "track_primary_outcomes", "track_secondary_outcomes", 
                                "track_drug_route", "track_drug_class", "regime_drug_class", 
                                "latest_company_approval", "ceo_name", "patent_expiration_date", 
                                "exclusivity_Date", "Ingredient", "Applicant_Full_Name", 
                                "Patent_Expire_Date_Text", "Exclusivity_Date", "Open_on_Approval", "filled50", "filled121"], 
                        default="track_trial_ids",
                        help="Task to perform",
                        )   
    parser.add_argument("--search_context_size", choices=["low", "medium", "high"], default="medium",
                        help="Search context size for OpenAI Search models")
    parser.add_argument("--use_judge", action="store_true",
                        help="Use LLM as judge for evaluating correctness instead of regex/rules.")
    parser.add_argument("--judge_model", default="gpt-4.1-mini",
                        help="OpenAI model to use for the LLM judge.")

    args = parser.parse_args()
    
    if args.model not in ALL_MODELS:
        logger.error(f"Invalid model name: {args.model}")
        logger.error(f"Available models: {', '.join(ALL_MODELS.keys())}")
        return
    
    # Select the correct inference backend
    if args.model in GEMINI_MODELS:
        run_inference = gemini_run_inference_multithread
        inference_kwargs = {"model_name": args.model}
    elif args.model in SONAR_MODELS:
        run_inference = sonar_run_inference_multithread
        inference_kwargs = {"model_name": args.model}
    elif args.model in OPENAI_SEARCH_MODELS:
        run_inference = openai_search_run_inference_multithread
        inference_kwargs = {"model_name": args.model, "web_search_options": {"search_context_size": args.search_context_size}}
    else:
        logger.error(f"Model {args.model} not recognized.")
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
        task=args.task,
        run_inference=run_inference,
        inference_kwargs=inference_kwargs,
        use_judge=args.use_judge,
        judge_model=args.judge_model
    )

if __name__ == "__main__":
    main()
