import base64
import os
import time
import json
import random
import logging
import csv
import io
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union, Generator, Tuple, Any
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GeminiInference")

# Load environment variables from .env file
load_dotenv()

# Define available Gemini models
GEMINI_MODELS = {
    "gemini-1.0-pro": "gemini-1.0-pro",
    "gemini-1.0-pro-vision": "gemini-1.0-pro-vision",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-2.0-pro": "gemini-2.0-pro",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.5-pro-exp-03-25": "gemini-2.5-pro-exp-03-25",
}

class ExponentialBackoff:
    """
    Implements exponential backoff strategy for API retries
    """
    def __init__(
        self, 
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        max_retries: int = 5,
        jitter: bool = True,
        backoff_factor: float = 2.0
    ):
        """
        Initialize exponential backoff parameters
        
        Args:
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            max_retries: Maximum number of retry attempts
            jitter: Whether to add randomness to delay
            backoff_factor: Multiplier for backoff calculation
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.jitter = jitter
        self.backoff_factor = backoff_factor
        self.retry_count = 0
    
    def reset(self):
        """Reset retry counter"""
        self.retry_count = 0
    
    def get_next_delay(self) -> Optional[float]:
        """
        Calculate next delay time using exponential backoff
        
        Returns:
            Delay time in seconds, or None if max retries reached
        """
        if self.retry_count >= self.max_retries:
            return None
        
        # Calculate delay with exponential backoff
        delay = min(
            self.max_delay,
            self.initial_delay * (self.backoff_factor ** self.retry_count)
        )
        
        # Add jitter if enabled (0.8-1.2 times the delay)
        if self.jitter:
            delay = delay * (0.8 + random.random() * 0.4)
            
        self.retry_count += 1
        return delay

class GeminiInference:
    def __init__(
        self, 
        model_name: str = "gemini-2.0-flash",
        backoff_config: Optional[Dict] = None
    ):
        """
        Initialize Gemini inference with specified model
        
        Args:
            model_name: Name of the Gemini model to use
            backoff_config: Configuration for exponential backoff
        """
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please add it to .env file.")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        if model_name not in GEMINI_MODELS:
            available_models = ", ".join(GEMINI_MODELS.keys())
            raise ValueError(f"Invalid model name: {model_name}. Available models: {available_models}")
            
        self.model_name = GEMINI_MODELS[model_name]
        self.model = genai.GenerativeModel(self.model_name)
        
        # Initialize backoff strategy
        backoff_defaults = {
            "initial_delay": 1.0,
            "max_delay": 60.0,
            "max_retries": 5,
            "jitter": True,
            "backoff_factor": 2.0
        }
        backoff_config = backoff_config or backoff_defaults
        self.backoff = ExponentialBackoff(**backoff_config)
    
    def generate_response(
        self, 
        input_text: str, 
        use_tools: bool = False, 
        stream: bool = False,
        save_search_results: bool = True
    ) -> Union[str, Tuple[str, Optional[List[Dict]]], Generator]:
        """
        Generate response from Gemini model with exponential backoff on failure
        
        Args:
            input_text: Input text to send to the model
            use_tools: Whether to enable Google Search tool
            stream: Whether to stream the response
            save_search_results: Whether to save search results data
            
        Returns:
            Model response as string, or tuple of (response, search_results) if save_search_results=True,
            or generator if streaming
        """
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Configure model with or without tools
        self._configure_model(generation_config, safety_settings, use_tools)
        
        # Reset backoff for new request
        self.backoff.reset()
        search_results = []
        
        while True:
            try:
                if stream:
                    return self._stream_response(input_text)
                else:
                    response = self._get_response(input_text)
                    logger.info(f"Debug - Response type: {type(response)}")
                    
                    # Extract search results if tools were used
                    if use_tools and save_search_results and hasattr(response, 'candidates'):
                        search_results = self._extract_search_results(response)
                    
                    response_text = self._extract_text(response)
                    
                    # Return just the text if we're not saving search results
                    if not save_search_results:
                        return response_text
                    else:
                        return response_text, search_results
                
            except Exception as e:
                # Get next delay from backoff strategy
                delay = self.backoff.get_next_delay()
                
                if delay is None:
                    # Max retries reached, return error
                    error_msg = f"Error generating response after max retries: {str(e)}"
                    logger.error(error_msg)
                    if save_search_results:
                        return error_msg, None
                    else:
                        return error_msg
                
                # Log retry attempt
                logger.warning(
                    f"API call failed: {str(e)}. Retrying in {delay:.2f} seconds. "
                    f"(Attempt {self.backoff.retry_count}/{self.backoff.max_retries})"
                )
                
                # Wait before retrying
                time.sleep(delay)
    
    def _configure_model(self, generation_config: Dict, safety_settings: Dict, use_tools: bool):
        """Configure the model with appropriate settings"""
        if use_tools:
            # Enable Google Search
            tools = [
                {
                    "function_declarations": [
                        {
                            "name": "google_search",
                            "description": "Search Google for relevant information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    ]
                }
            ]
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools
            )
        else:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
    
    def _get_response(self, input_text) -> Any:
        """Get complete response at once"""
        # Check if input is a tuple/list that might be causing the CSV error
        if isinstance(input_text, (tuple, list)):
            logger.warning(f"Input is a {type(input_text).__name__} type, converting to string")
            logger.info(f"Debug - Input structure: {str(input_text)[:200]}")
            
            # If it's a structured data that might be CSV, join it properly
            if all(isinstance(item, (tuple, list)) for item in input_text):
                # Convert nested list/tuple to CSV string
                logger.info(f"Debug - Converting nested structure to CSV")
                csv_rows = []
                for row in input_text:
                    csv_rows.append(','.join(str(item) for item in row))
                input_text = '\n'.join(csv_rows)
                logger.info(f"Debug - Converted to CSV string (first 100 chars): {input_text[:100]}")
            else:
                # Simple list/tuple
                logger.info(f"Debug - Converting simple list to newline-separated string")
                input_text = '\n'.join(str(item) for item in input_text)
        
        # Get response and log its structure
        logger.info(f"Debug - Sending to model, input type: {type(input_text)}")
        response = self.model.generate_content(input_text)
        
        # Log response structure
        logger.info(f"Debug - Response type: {type(response)}")
        
        # Try to access common attributes for debugging
        if hasattr(response, 'text'):
            logger.info(f"Debug - Response has 'text' attribute: {response.text[:100] if len(response.text) > 100 else response.text}")
        if hasattr(response, 'parts'):
            logger.info(f"Debug - Response has 'parts' attribute, length: {len(response.parts)}")
        if hasattr(response, 'candidates'):
            logger.info(f"Debug - Response has 'candidates' attribute, length: {len(response.candidates)}")
        
        return response
    
    def _extract_text(self, response) -> str:
        """Extract text from response object with debug info"""
        logger.info(f"Debug - Extracting text from response type: {type(response)}")
        
        if hasattr(response, 'text'):
            logger.info(f"Debug - Using response.text attribute")
            return response.text
        elif hasattr(response, 'parts') and response.parts:
            logger.info(f"Debug - Using response.parts[0].text, parts length: {len(response.parts)}")
            if len(response.parts) > 0 and hasattr(response.parts[0], 'text'):
                return response.parts[0].text
            else:
                logger.info(f"Debug - parts[0] format: {dir(response.parts[0]) if response.parts else 'no parts'}")
                return str(response.parts)
        elif hasattr(response, 'candidates') and response.candidates:
            logger.info(f"Debug - Using response.candidates approach")
            candidates_data = []
            for i, candidate in enumerate(response.candidates):
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for j, part in enumerate(candidate.content.parts):
                        candidates_data.append(f"Candidate {i}, Part {j}: {str(part)[:100]}")
            logger.info(f"Debug - Candidates data: {candidates_data}")
            
            # Try to extract text from first candidate's content
            if response.candidates and hasattr(response.candidates[0], 'content'):
                content = response.candidates[0].content
                if hasattr(content, 'parts') and content.parts and hasattr(content.parts[0], 'text'):
                    return content.parts[0].text
        
        # Fallback to string representation
        logger.info(f"Debug - Falling back to str(response)")
        return str(response)
    
    def _extract_search_results(self, response) -> List[Dict]:
        """Extract search results from response with tool calls"""
        search_results = []
        
        try:
            if not hasattr(response, 'candidates') or not response.candidates:
                return search_results
                
            for candidate in response.candidates:
                if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
                    continue
                    
                for part in candidate.content.parts:
                    if not hasattr(part, 'function_call'):
                        continue
                        
                    # Extract function calls
                    if part.function_call and part.function_call.name == "google_search":
                        # Get query - handle MapComposite objects safely
                        if hasattr(part.function_call, 'args'):
                            # Handle different arg types
                            if isinstance(part.function_call.args, str):
                                try:
                                    args_dict = json.loads(part.function_call.args)
                                    query = args_dict.get("query", "")
                                except json.JSONDecodeError:
                                    query = str(part.function_call.args)
                            elif hasattr(part.function_call.args, 'query'):
                                # Handle MapComposite type
                                query = str(part.function_call.args.query)
                            else:
                                # Fallback to string representation
                                query = str(part.function_call.args)
                        else:
                            query = "Unknown query"
                        
                        # Get response if available
                        if hasattr(part, 'function_response') and part.function_response:
                            try:
                                # Handle different response formats
                                if hasattr(part.function_response, 'parts') and part.function_response.parts:
                                    result_text = part.function_response.parts[0].text
                                    try:
                                        result_data = json.loads(result_text)
                                    except json.JSONDecodeError:
                                        # Store as plain text if not valid JSON
                                        result_data = {"raw_text": result_text}
                                else:
                                    # Store response object attributes as dictionary
                                    result_data = self._object_to_dict(part.function_response)
                                
                                search_results.append({
                                    "query": query,
                                    "results": result_data
                                })
                            except (AttributeError, IndexError) as e:
                                logger.warning(f"Failed to parse search results: {str(e)}")
        
        except Exception as e:
            logger.warning(f"Error extracting search results: {str(e)}")
            
        return search_results
        
    def _object_to_dict(self, obj) -> Dict:
        """Convert object attributes to dictionary safely"""
        if obj is None:
            return {}
            
        result = {}
        # Get all attributes that don't start with underscore
        for attr in dir(obj):
            if not attr.startswith('_'):
                try:
                    value = getattr(obj, attr)
                    # Skip methods and complex objects
                    if not callable(value):
                        # Convert nested objects recursively
                        if hasattr(value, '__dict__'):
                            result[attr] = self._object_to_dict(value)
                        # Handle lists of objects
                        elif isinstance(value, list):
                            result[attr] = [
                                self._object_to_dict(item) if hasattr(item, '__dict__') else item 
                                for item in value
                            ]
                        # Basic types can be added directly
                        else:
                            try:
                                # Try to convert to JSON-serializable format
                                json.dumps({attr: value})
                                result[attr] = value
                            except (TypeError, OverflowError):
                                # If not serializable, use string representation
                                result[attr] = str(value)
                except Exception as e:
                    result[attr] = f"<Error accessing attribute: {str(e)}>"
                    
        return result
    
    def _stream_response(self, input_text) -> Generator:
        """Stream response chunks"""
        response_stream = self.model.generate_content(input_text, stream=True)
        
        for chunk in response_stream:
            if hasattr(chunk, 'text'):
                yield chunk.text
            elif hasattr(chunk, 'parts') and chunk.parts:
                yield chunk.parts[0].text
    
    def save_search_results(self, search_results: List[Dict], filename: str = "search_results.json"):
        """
        Save search results to a JSON file
        
        Args:
            search_results: List of search result dictionaries
            filename: Output JSON filename
        """
        if not search_results:
            logger.warning("No search results to save")
            return
            
        try:
            with open(filename, 'w') as f:
                json.dump(search_results, f, indent=2)
            logger.info(f"Search results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save search results: {str(e)}")

def run_inference_multithread(
    model_name: str,
    input_list: List[str],
    use_tools: bool = False,
    save_search_results: bool = True,
    max_workers: int = 4,
    backoff_config: Optional[Dict] = None
) -> List[Union[str, Tuple[str, Optional[List[Dict]]]]]:
    """
    Run inference on multiple inputs in parallel using multithreading
    
    Args:
        model_name: Name of the Gemini model
        input_list: List of input texts
        use_tools: Whether to enable Google Search tool
        save_search_results: Whether to save search results data
        max_workers: Maximum number of threads
        backoff_config: Configuration for exponential backoff
        
    Returns:
        List of responses (strings if save_search_results=False, tuples if save_search_results=True)
    """
    def process_input(input_text):
        inference = GeminiInference(model_name=model_name, backoff_config=backoff_config)
        result = inference.generate_response(
            input_text, 
            use_tools=use_tools,
            save_search_results=save_search_results
        )
        
        # Debug what's being returned
        logger.info(f"Debug - process_input result type: {type(result)}")
        
        # Always ensure we return just the text part if we got a tuple but don't need search results
        if not save_search_results and isinstance(result, tuple):
            logger.info(f"Debug - Extracting text from tuple result since save_search_results=False")
            return result[0]  # Return just the text response, not the tuple
        
        return result
    
    logger.info(f"Running inference on {len(input_list)} examples with {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and track with tqdm progress bar
        futures = [executor.submit(process_input, text) for text in input_list]
        results = []
        
        # Process results as they complete with progress bar
        for f in tqdm(futures, total=len(futures), desc="Model inference"):
            try:
                result = f.result()
                logger.info(f"Debug - future.result() type: {type(result)}")
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
                logger.error(traceback.format_exc())
                # Add an error message as the result
                results.append(f"ERROR: {str(e)}")
    
    return results

def process_csv_data(data):
    """
    Safely process CSV data regardless of input type
    
    Args:
        data: CSV data which could be string, bytes, file-like object, or response from API
        
    Returns:
        List of dictionaries representing CSV rows with headers as keys
    """
    logger.info(f"Debug - process_csv_data input type: {type(data)}")
    
    try:
        # Handle various input types
        if isinstance(data, (list, tuple)):
            # If it's already a list/tuple, check if it's a list of rows
            if data and isinstance(data[0], (list, tuple, dict)):
                if isinstance(data[0], dict):
                    return data  # Already in the right format
                else:
                    # Convert list of lists to list of dicts
                    headers = data[0]
                    logger.info(f"Debug - Converting list of lists with headers: {headers}")
                    return [dict(zip(headers, row)) for row in data[1:]]
            else:
                # It's a single row, convert to string
                logger.info(f"Debug - Converting single row to CSV string")
                csv_str = ','.join(str(item) for item in data)
        elif hasattr(data, 'read'):
            # File-like object
            logger.info(f"Debug - Reading from file-like object")
            csv_str = data.read()
            if isinstance(csv_str, bytes):
                csv_str = csv_str.decode('utf-8')
        elif isinstance(data, bytes):
            # Bytes object
            logger.info(f"Debug - Decoding bytes to string")
            csv_str = data.decode('utf-8')
        elif isinstance(data, str):
            # String
            logger.info(f"Debug - Using input as CSV string")
            csv_str = data
        else:
            # Before giving up, check if it's a response object from the API
            logger.info(f"Debug - Checking if it's a response object: {dir(data)[:200] if hasattr(data, '__dir__') else 'No dir method'}")
            
            # Check for common response object attributes
            if hasattr(data, 'text'):
                csv_str = data.text
            elif hasattr(data, 'parts') and data.parts:
                # Get text from first part
                csv_str = data.parts[0].text if hasattr(data.parts[0], 'text') else str(data.parts[0])
            elif hasattr(data, 'candidates') and data.candidates:
                # Try to get text from first candidate
                candidate = data.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    csv_str = candidate.content.parts[0].text
                else:
                    csv_str = str(candidate)
            else:
                # Try to convert to string as last resort
                csv_str = str(data)
                logger.warning(f"Debug - Converted unknown type to string: {csv_str[:100]}")
        
        # Process CSV string - add an extra safeguard
        logger.info(f"Debug - CSV string to parse (first 100 chars): {csv_str[:100] if isinstance(csv_str, str) else 'Not a string'}")
        
        # Make absolutely sure we have a string
        if not isinstance(csv_str, str):
            raise TypeError(f"Expected string for CSV parsing, got {type(csv_str)}")
            
        # Process CSV string
        reader = csv.DictReader(io.StringIO(csv_str))
        result = list(reader)
        logger.info(f"Debug - Successfully parsed {len(result)} rows")
        return result
        
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        logger.error(traceback.format_exc())
        if isinstance(data, (list, tuple)) and data:
            logger.error(f"First item type: {type(data[0])}")
            logger.error(f"First few items: {str(data[:2])}")
        elif hasattr(data, '__dict__'):
            logger.error(f"Object attributes: {dir(data)}")
        raise

def save_batch_results(
    results: List[Union[str, Tuple[str, Optional[List[Dict]]]]],
    output_dir: str = "results",
    save_search_results: bool = True
):
    """
    Save batch inference results to files
    
    Args:
        results: List of responses or (response, search_results) tuples
        output_dir: Directory to save results
        save_search_results: Whether search results were saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save responses
    responses = []
    for i, result in enumerate(results):
        if save_search_results and isinstance(result, tuple):
            response, search_data = result
            responses.append(response)
            
            # Save individual search results
            if search_data:
                search_file = os.path.join(output_dir, f"search_results_{i+1}.json")
                with open(search_file, 'w') as f:
                    json.dump(search_data, f, indent=2)
        else:
            responses.append(result)
    
    # Save all responses to a single file
    with open(os.path.join(output_dir, "responses.txt"), 'w') as f:
        for i, response in enumerate(responses):
            f.write(f"=== Response {i+1} ===\n")
            f.write(response)
            f.write("\n\n")
    
    logger.info(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    # Example usage
    inference = GeminiInference()
    
    # Example without tools
    print("Generating response without tools:")
    response = inference.generate_response("What is multiple myeloma?", use_tools=False, save_search_results=False)
    print(response)
    print("\n" + "-"*80 + "\n")
    
    # Example with tools and search result saving
    print("Generating response with tools:")
    response, search_results = inference.generate_response(
        "What is the latest treatment for multiple myeloma?", 
        use_tools=True,
        save_search_results=True
    )
    print(response)
    
    # Save search results
    if search_results:
        inference.save_search_results(search_results, "myeloma_search_results.json")
        print(f"Search results saved to myeloma_search_results.json")
    print("\n" + "-"*80 + "\n")
    
    # Example with streaming
    print("Streaming response:")
    for chunk in inference.generate_response("What are common symptoms of multiple myeloma?", stream=True):
        print(chunk, end="")
    print("\n" + "-"*80 + "\n")
    
    # Example demonstrating CSV data handling
    print("Handling CSV data:")
    # Sample CSV data as a tuple (which would cause an error without our fix)
    csv_data = (
        ("Patient", "Age", "Diagnosis"),
        ("P001", "62", "Multiple Myeloma"),
        ("P002", "58", "MGUS"),
        ("P003", "71", "Multiple Myeloma")
    )
    
    # Process the CSV data properly
    print("CSV data processed:")
    processed_data = process_csv_data(csv_data)
    for row in processed_data:
        print(row)
    print("\n" + "-"*80 + "\n")
    
    # Example with multithreading
    print("Multithreaded inference:")
    inputs = [
        "What is multiple myeloma?",
        "What are the symptoms of multiple myeloma?",
        "How is multiple myeloma diagnosed?"
    ]
    
    # Configure custom backoff strategy
    backoff_config = {
        "initial_delay": 2.0,
        "max_delay": 30.0,
        "max_retries": 3,
        "jitter": True,
        "backoff_factor": 1.5
    }
    
    results = run_inference_multithread(
        "gemini-2.0-flash", 
        inputs, 
        use_tools=True,
        save_search_results=True,
        backoff_config=backoff_config
    )
    
    # Save batch results
    save_batch_results(results, "batch_results", save_search_results=True)
    
    # Display results
    for i, result in enumerate(results):
        if isinstance(result, tuple):
            response, search_data = result
            print(f"Result {i+1}:")
            print(response)
            print(f"Search data captured: {bool(search_data)}")
        else:
            print(f"Result {i+1}:")
            print(result)
            print(f"No search data captured")
        print("-" * 40)