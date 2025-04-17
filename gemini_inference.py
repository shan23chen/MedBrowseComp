import base64
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union, Generator
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Define available Gemini models
GEMINI_MODELS = {
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.5-pro-preview-03-25": "gemini-2.5-pro-preview-03-25"
}

class GeminiInference:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize Gemini inference with specified model
        
        Args:
            model_name: Name of the Gemini model to use
        """
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please add it to .env file.")
        
        self.client = genai.Client(api_key=self.api_key)
        
        if model_name not in GEMINI_MODELS:
            available_models = ", ".join(GEMINI_MODELS.keys())
            raise ValueError(f"Invalid model name: {model_name}. Available models: {available_models}")
            
        self.model_name = GEMINI_MODELS[model_name]
        
        # Backoff parameters
        self.max_retries = 5
        self.initial_backoff = 1  # seconds
        self.max_backoff = 60  # seconds
        self.backoff_factor = 2  # exponential factor
        self.jitter = 0.1  # adds randomness to avoid thundering herd
    
    def _backoff_time(self, retry: int) -> float:
        """Calculate backoff time with jitter for a given retry attempt"""
        backoff = min(self.max_backoff, self.initial_backoff * (self.backoff_factor ** retry))
        # Add jitter: random value between -10% and +10% of the backoff
        jitter_amount = backoff * self.jitter
        backoff = backoff + random.uniform(-jitter_amount, jitter_amount)
        return max(0, backoff)  # Ensure we don't get negative backoff
    
    def generate_response(self, 
                          input_text: str, 
                          use_tools: bool = False, 
                          stream: bool = False) -> Union[str, Dict, Generator]:
        """
        Generate response from Gemini model with exponential backoff for 429 errors
        
        Args:
            input_text: Input text to send to the model
            use_tools: Whether to enable Google Search tool
            stream: Whether to stream the response
            
        Returns:
            If use_tools=False: Model response as string or generator if streaming
            If use_tools=True: Dictionary with 'text' and 'citations' keys or generator if streaming
        """
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=input_text)],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )
        
        if use_tools:
            generate_content_config.tools = [types.Tool(google_search=types.GoogleSearch())]
        
        for retry in range(self.max_retries):
            try:
                if stream:
                    return self._stream_response_with_backoff(contents, generate_content_config, use_tools)
                else:
                    return self._get_response_with_backoff(contents, generate_content_config, use_tools)
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a 429 error (rate limit)
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    if retry < self.max_retries - 1:  # Don't sleep on the last retry
                        backoff_time = self._backoff_time(retry)
                        print(f"Rate limit exceeded. Retrying in {backoff_time:.2f} seconds (attempt {retry+1}/{self.max_retries})...")
                        time.sleep(backoff_time)
                    else:
                        return f"Error: Rate limit exceeded after {self.max_retries} retries. Please try again later."
                else:
                    # Not a rate limit error, return immediately
                    return f"Error generating response: {str(e)}"
        
        # If we get here, we've exhausted all retries
        return "Error: Maximum retries exceeded. Please try again later."
    
    def _get_response_with_backoff(self, contents, config, use_tools) -> Union[str, Dict]:
        """Get complete response at once with backoff handling"""
        for retry in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                
                if use_tools and hasattr(response, 'candidates') and response.candidates:
                    citations = []
                    # Extract citation information from tool use
                    for candidate in response.candidates:
                        if hasattr(candidate, 'citation_metadata') and candidate.citation_metadata:
                            for citation in candidate.citation_metadata.citations:
                                if hasattr(citation, 'uri'):
                                    citations.append(citation.uri)
                                elif hasattr(citation, 'url'):  # Some versions might use url instead of uri
                                    citations.append(citation.url)
                                    
                    return {
                        'text': response.text,
                        'citations': citations
                    }
                
                return response.text
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a 429 error (rate limit)
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    if retry < self.max_retries - 1:  # Don't sleep on the last retry
                        backoff_time = self._backoff_time(retry)
                        print(f"Rate limit exceeded. Retrying in {backoff_time:.2f} seconds (attempt {retry+1}/{self.max_retries})...")
                        time.sleep(backoff_time)
                    else:
                        raise Exception(f"Rate limit exceeded after {self.max_retries} retries")
                else:
                    # Not a rate limit error, raise immediately
                    raise
    
    def _stream_response_with_backoff(self, contents, config, use_tools) -> Generator:
        """Stream response chunks with backoff handling"""
        for retry in range(self.max_retries):
            try:
                response_stream = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                
                if use_tools:
                    all_text = ""
                    citations = []
                    
                    for chunk in response_stream:
                        if chunk.text:
                            all_text += chunk.text
                            yield chunk.text
                            
                        # Try to extract citation information if available in the chunk
                        if hasattr(chunk, 'candidates') and chunk.candidates:
                            for candidate in chunk.candidates:
                                if hasattr(candidate, 'citation_metadata') and candidate.citation_metadata:
                                    for citation in candidate.citation_metadata.citations:
                                        uri = None
                                        if hasattr(citation, 'uri'):
                                            uri = citation.uri
                                        elif hasattr(citation, 'url'):
                                            uri = citation.url
                                        
                                        if uri and uri not in citations:
                                            citations.append(uri)
                    
                    # After streaming is complete, yield a final dictionary with all text and citations
                    yield {
                        'text': all_text,
                        'citations': citations
                    }
                else:
                    for chunk in response_stream:
                        if chunk.text:
                            yield chunk.text
                
                # If we get here without errors, we're done
                return
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a 429 error (rate limit)
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    if retry < self.max_retries - 1:  # Don't sleep on the last retry
                        backoff_time = self._backoff_time(retry)
                        print(f"Rate limit exceeded during streaming. Retrying in {backoff_time:.2f} seconds (attempt {retry+1}/{self.max_retries})...")
                        time.sleep(backoff_time)
                        # On retry, we'll restart the stream from the beginning
                    else:
                        yield f"Error: Rate limit exceeded after {self.max_retries} retries. Please try again later."
                        return
                else:
                    # Not a rate limit error, yield the error and return
                    yield f"Error during streaming: {str(e)}"
                    return
        
        # If we get here, we've exhausted all retries
        yield "Error: Maximum retries exceeded. Please try again later."

def run_inference_multithread(
    model_name: str,
    input_list: List[str],
    use_tools: bool = False,
    max_workers: int = 4
) -> List[Union[str, Dict]]:
    """
    Run inference on multiple inputs in parallel using multithreading
    
    Args:
        model_name: Name of the Gemini model
        input_list: List of input texts
        use_tools: Whether to enable Google Search tool
        max_workers: Maximum number of threads
        
    Returns:
        List of responses (strings or dictionaries with text and citations)
    """
    results = []
    
    def process_input(input_text):
        inference = GeminiInference(model_name=model_name)
        return inference.generate_response(input_text, use_tools=use_tools)
    
    print(f"Running inference on {len(input_list)} examples with {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and track with tqdm progress bar
        futures = [executor.submit(process_input, text) for text in input_list]
        results = []
        
        # Process results as they complete with progress bar
        for f in tqdm(futures, total=len(futures), desc="Model inference"):
            results.append(f.result())
    
    return results

if __name__ == "__main__":
    # Example usage
    inference = GeminiInference()
    
    # Example without tools
    print("Generating response without tools:")
    response = inference.generate_response("What is multiple myeloma?", use_tools=False)
    print(response)
    print("\n" + "-"*80 + "\n")
    
    # Example with tools
    print("Generating response with tools:")
    response = inference.generate_response("What is the latest treatment for multiple myeloma?", use_tools=True)
    if isinstance(response, dict):
        print("Response text:")
        print(response['text'])
        print("\nCitations:")
        for url in response['citations']:
            print(f"- {url}")
    else:
        print(response)
    print("\n" + "-"*80 + "\n")
    
    # Example with streaming
    print("Streaming response:")
    last_response = None
    for chunk in inference.generate_response("What are common symptoms of multiple myeloma?", stream=True, use_tools=True):
        if isinstance(chunk, dict):
            last_response = chunk  # Store the final chunk with citations
        else:
            print(chunk, end="")
    
    print("\n")
    if last_response and isinstance(last_response, dict) and 'citations' in last_response:
        print("\nCitations:")
        for url in last_response['citations']:
            print(f"- {url}")
    print("\n" + "-"*80 + "\n")
    
    # Example with multithreading
    print("Multithreaded inference:")
    inputs = [
        "What is multiple myeloma?",
        "What are the symptoms of multiple myeloma?",
        "How is multiple myeloma diagnosed?"
    ]
    
    results = run_inference_multithread("gemini-2.0-flash", inputs, use_tools=True)
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        if isinstance(result, dict):
            print(result['text'])
            print("\nCitations:")
            for url in result['citations']:
                print(f"- {url}")
        else:
            print(result)
        print("-" * 40)