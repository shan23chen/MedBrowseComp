import base64
import os
import time
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
    "gemini-1.0-pro": "gemini-1.0-pro",
    "gemini-1.0-pro-vision": "gemini-1.0-pro-vision",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-2.0-pro": "gemini-2.0-pro",
    "gemini-2.0-flash": "gemini-2.0-flash",
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
    
    def generate_response(self, 
                          input_text: str, 
                          use_tools: bool = False, 
                          stream: bool = False) -> Union[str, Generator]:
        """
        Generate response from Gemini model
        
        Args:
            input_text: Input text to send to the model
            use_tools: Whether to enable Google Search tool
            stream: Whether to stream the response
            
        Returns:
            Model response as string or generator if streaming
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
        
        try:
            if stream:
                return self._stream_response(contents, generate_content_config)
            else:
                return self._get_response(contents, generate_content_config)
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _get_response(self, contents, config) -> str:
        """Get complete response at once"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        
        return response.text
    
    def _stream_response(self, contents, config) -> Generator:
        """Stream response chunks"""
        response_stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

def run_inference_multithread(
    model_name: str,
    input_list: List[str],
    use_tools: bool = False,
    max_workers: int = 4
) -> List[str]:
    """
    Run inference on multiple inputs in parallel using multithreading
    
    Args:
        model_name: Name of the Gemini model
        input_list: List of input texts
        use_tools: Whether to enable Google Search tool
        max_workers: Maximum number of threads
        
    Returns:
        List of responses
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
    print(response)
    print("\n" + "-"*80 + "\n")
    
    # Example with streaming
    print("Streaming response:")
    for chunk in inference.generate_response("What are common symptoms of multiple myeloma?", stream=True):
        print(chunk, end="")
    print("\n" + "-"*80 + "\n")
    
    # Example with multithreading
    print("Multithreaded inference:")
    inputs = [
        "What is multiple myeloma?",
        "What are the symptoms of multiple myeloma?",
        "How is multiple myeloma diagnosed?"
    ]
    
    results = run_inference_multithread("gemini-2.0-flash", inputs)
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(result)
        print("-" * 40)
