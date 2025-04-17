import os
from typing import List, Union, Dict, Generator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_SEARCH_MODELS = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.1": "gpt-4.1",
    # Add more as needed
}

class OpenAISearchInference:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please add it to .env file.")
        if model_name not in OPENAI_SEARCH_MODELS:
            raise ValueError(f"Invalid model name: {model_name}. Available models: {', '.join(OPENAI_SEARCH_MODELS.keys())}")
        self.model_name = OPENAI_SEARCH_MODELS[model_name]
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, input_text: str, stream: bool = False, web_search_options: dict = None) -> Union[str, Dict]:
        """
        Generates a response from the selected OpenAI model using the Responses API and web_search_preview tool.
        This is used for all supported models, including gpt-4o-mini, gpt-4o-search-preview, gpt-4.1, etc.
        """
        # Default tool config
        tool_config = {
            "type": "web_search_preview",
            "user_location": {
                "type": "approximate"
            },
            "search_context_size": "medium"
        }
        # Allow user to override tool config
        if web_search_options:
            tool_config.update(web_search_options)
        payload = {
            "model": self.model_name,  # Pass as given (e.g., "gpt-4o-mini")
            "input": [{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": input_text
                    }
                ]
            }],
            "text": {
                "format": {
                    "type": "text"
                }
            },
            "reasoning": {},
            "tools": [tool_config],
            "temperature": 1,
            "max_output_tokens": 10000,
            "top_p": 1,
            "store": True
        }
        response = self.client.responses.create(**payload)
        # Debug: print the response structure
        # print("[DEBUG] OpenAI responses.create raw response:", response)
        # Attempt to extract the text output from the response object
        try:
            # response.output is a list, look for ResponseOutputMessage
            for output_obj in getattr(response, "output", []):
                # ResponseOutputMessage has .content which is a list
                if hasattr(output_obj, "content"):
                    for content_obj in getattr(output_obj, "content", []):
                        # ResponseOutputText has .text
                        if hasattr(content_obj, "text"):
                            return content_obj.text
            # If nothing found, fallback to string representation
            return str(response)
        except Exception as e:
            print(f"[ERROR] Failed to extract text from response: {e}")
            return str(response)



def run_inference_multithread(model_name: str, input_list: List[str], max_workers: int = 4, web_search_options: dict = None) -> List[Union[str, Dict]]:
    inference = OpenAISearchInference(model_name)
    def _inference(input_text):
        return inference.generate_response(input_text, web_search_options=web_search_options)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_inference, text) for text in input_list]
        results = []
        for f in tqdm(futures, total=len(futures), desc="Model inference"):
            results.append(f.result())
    return results
