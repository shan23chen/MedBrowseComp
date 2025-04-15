import os
from typing import List, Union, Dict, Generator
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SONAR_MODELS = {
    "sonar-deep-research": "sonar-deep-research",
    "sonar-reasoning-pro": "sonar-reasoning-pro",
    "sonar-reasoning": "sonar-reasoning",
    "sonar-pro": "sonar-pro",
    "sonar": "sonar",
}

class SonarInference:
    def __init__(self, model_name: str = "sonar-pro"):
        self.api_key = os.environ.get("SONAR_API_KEY")
        if not self.api_key:
            raise ValueError("SONAR_API_KEY environment variable not set. Please add it to .env file.")
        if model_name not in SONAR_MODELS:
            raise ValueError(f"Invalid model name: {model_name}. Available models: {', '.join(SONAR_MODELS.keys())}")
        self.model_name = SONAR_MODELS[model_name]
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.perplexity.ai")

    def generate_response(self, input_text: str, stream: bool = False) -> Union[str, Generator]:
        messages = [
            {"role": "system", "content": "You are an artificial intelligence assistant and you need to engage in a helpful, detailed, polite conversation with a user."},
            {"role": "user", "content": input_text},
        ]
        if stream:
            response_stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
            )
            return (resp.choices[0].message.content for resp in response_stream)
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            return response.choices[0].message.content

def run_inference_multithread(model_name: str, input_list: List[str], max_workers: int = 4) -> List[Union[str, Dict]]:
    inference = SonarInference(model_name)
    def _inference(input_text):
        return inference.generate_response(input_text)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_inference, input_list))
    return results
