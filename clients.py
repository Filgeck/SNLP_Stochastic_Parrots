from collections import deque
from datetime import datetime, timedelta
import time
import anthropic
import ollama
import os
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold
from datasets import load_dataset
from typing import List, Callable
import traceback
import dotenv
from openai import OpenAI

# Load environment variables from .envrc file
# put your API keys here
dotenv.load_dotenv(".envrc")


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class Retries:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def _func_with_retries[R](self, func: Callable[..., R], *args, **kwargs) -> R:
        for num_retries in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                print(
                    f"Failed attempt {num_retries + 1} of {self.max_retries} running {func}"
                )
                if num_retries < self.max_retries - 1:
                    print("Retrying due to error:")
                    print(traceback.format_exc())
                else:
                    print("Max retries reached. Error:")
                    print(traceback.format_exc())
        raise RuntimeError(f"Function {func} failed after maximum retries")


class ModelClient(Retries):

    def __init__(self, model_name: str, max_retries: int = 3):
        super().__init__(max_retries=max_retries)
        self.model_name = model_name
        self.client: OpenAI | None = None
        self._request_limit_per_minute: float | None = None
        self.is_openrouter = False
        if self.model_name.startswith("anthropic"):
            self.is_openrouter = True
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
        self._request_history = deque[dict]()

    def query(self, prompt: str) -> str:
        if self.is_openrouter:
            return self._func_with_retries(self._query_openrouter, prompt)
        elif self.model_name in {"llama3.2", "deepseek-r1:8b"}:
            return self._func_with_retries(self._query_local_ollama, prompt)
        elif self.model_name in {"gemini-2.5-pro-exp-03-25", "gemini-1.5-pro"}:
            return self._func_with_retries(self._query_gemini, prompt)
        else:
            raise ValueError(f"Model {self.model_name} not supported")

    def _query_local_ollama(self, prompt: str) -> str:
        client = ollama.Client()

        response = client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        response_content = ""
        for chunk in response:
            content = chunk.message.content
            if content is None:
                raise ValueError(
                    f"Error: Received None content from model {self.model_name}"
                )
            response_content += content
            if len(response_content) > 50000:
                raise ValueError(
                    f"Error: Response from model {self.model_name} exceeded 50,000 characters. Likely model is repeating itself"
                )

        return response_content

    def _query_gemini(self, prompt: str) -> str:
        genai.configure(api_key=GOOGLE_API_KEY)

        generation_config = genai.GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=16384,
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        response: GenerateContentResponse = model.generate_content(
            prompt,
        )
        response_content = response.text

        return response_content
    
    def _query_openrouter(self, prompt: str) -> str:
        print("Querying OpenRouter model:", self.model_name)
        assert isinstance(self.client, OpenAI)
        completion = self.client.chat.completions.create(
            extra_body={
                "provider": {
                    "sort": "throughput"
                }
            },
            model=self.model_name,
            messages=[
                {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ]
                }
            ],
            n=1 # number of choices to generate
        )
        assert len(completion.choices) == 1
        message = completion.choices[0].message
        if message.content is None:
            raise ValueError(
                f"Error: Received None content from model {self.model_name}"
            )
        print("Received message")
        return message.content


class RagClient(Retries):
    def __init__(self, max_retries: int = 3):
        super().__init__(max_retries=max_retries)
        self.dataset = load_dataset("bigscience-data/roots_code_stackexchange")

    def query(self, prompt: str, num_retrieve: int = 10) -> List[str]:
        """Find the most similar problems to the given issue using precomputed embeddings."""

        # TODO: Implement

        # rag to find 10 most similar to issue,

        # then return in list

        raise NotImplementedError("RagClient query method not implemented yet")
