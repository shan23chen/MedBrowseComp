import os
from typing import List, Union, Dict, Generator
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_SEARCH_MODELS = {
    "gpt-4o-search-preview": "gpt-4o-search-preview",
    "gpt-4o-mini-search-preview": "gpt-4o-mini-search-preview",
    # Add more as needed
}

class OpenAISearchInference:
    def __init__(self, model_name: str = "gpt-4o-search-preview"):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please add it to .env file.")
        if model_name not in OPENAI_SEARCH_MODELS:
            raise ValueError(f"Invalid model name: {model_name}. Available models: {', '.join(OPENAI_SEARCH_MODELS.keys())}")
        self.model_name = OPENAI_SEARCH_MODELS[model_name]
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, input_text: str, stream: bool = False, web_search_options: dict = None) -> Union[str, Dict]:
        messages = [
            {"role": "user", "content": input_text},
        ]
        if web_search_options is None:
            web_search_options = {}
        response = self.client.chat.completions.create(
            model=self.model_name,
            web_search_options=web_search_options,
            messages=messages,
        )
        # Return the main message content string
        return response.choices[0].message.content

def run_inference_multithread(model_name: str, input_list: List[str], max_workers: int = 4, web_search_options: dict = None) -> List[Union[str, Dict]]:
    inference = OpenAISearchInference(model_name)
    def _inference(input_text):
        return inference.generate_response(input_text, web_search_options=web_search_options)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_inference, input_list))
    return results
