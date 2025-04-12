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

OLLAMA_MODELS = {"llama3.2", "deepseek-r1:8b"}
GEMINI_MODELS = {"gemini-2.5-pro-exp-03-25", "gemini-1.5-pro"}
ANTHROPIC_MODELS = {"claude-3-7-sonnet-20250219"}

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
        self._request_limit_per_minute: float | None = None
        if self.model_name in ANTHROPIC_MODELS:
            # For anthropic, limit is in tokens (~3.5 chars per token)
            # Actual limit is 20000 tokens per minute, but I'll use 17500 for
            # now
            # In this case, self._request_limit_per minute is in chars
            self._request_limit_per_minute = 17500 * 3.5
        else:
            self._request_limit_per_minute = None
        self._request_history = deque[dict]()
    
    def _get_requests_in_last_minute(self):
        """
        Calculate the total characters sent in the last minute.
        Also removes entries older than 1 minute from the history.
        
        Returns:
            int: Total characters sent in the last minute
        """
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Remove entries older than 1 minute
        while self._request_history and self._request_history[0]['timestamp'] < one_minute_ago:
            self._request_history.popleft()
        
        # Sum characters in the remaining entries
        return sum(entry['request_count'] for entry in self._request_history)

    def _wait_if_needed(self, prompt_length: int):
        while True:
            print("Waiting if needed...")
            requests_in_last_minute = self._get_requests_in_last_minute()
            
            # If adding this prompt would exceed the limit
            if requests_in_last_minute + prompt_length > self._request_limit_per_minute:
                # Calculate how long to wait until the oldest request drops off
                if self._request_history:
                    oldest_timestamp = self._request_history[0]['timestamp']
                    wait_time = (oldest_timestamp + timedelta(minutes=1) - datetime.now()).total_seconds()
                    
                    if wait_time > 0:
                        print(f"Rate limit would be exceeded. Waiting {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                        continue
                else:
                    # This should not happen normally, but just in case
                    print(f"requests_in_last_minute {requests_in_last_minute} + prompt_length {prompt_length} > self._request_limit_per_minute {self._request_limit_per_minute}, but self._request_history is empty. Waiting 1 second...")
                    time.sleep(1)
                    continue
            
            # If we get here, we're good to go
            break

    def query(self, prompt: str) -> str:
        if self.model_name in OLLAMA_MODELS:
            return self._func_with_retries(self._query_local_ollama, prompt)
        elif self.model_name in GEMINI_MODELS:
            return self._func_with_retries(self._query_gemini, prompt)
        elif self.model_name in ANTHROPIC_MODELS:
            return self._func_with_retries(self._query_claude, prompt)
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
    
    def _query_claude(self, prompt: str) -> str:

        client = anthropic.Anthropic()
        print("Query claude")
        print(f"Prompt len: {len(prompt)}")
        self._wait_if_needed(len(prompt))
        self._request_history.append({
            'timestamp': datetime.now(),
            'request_count': prompt
        })
        message = client.messages.create(
            model=self.model_name,
            max_tokens=16384,
            temperature=1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        response_contents = []
        for chunk in message.content:
            assert chunk.type == "text"
            response_contents.append(chunk.text)
        return "".join(response_contents)


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
