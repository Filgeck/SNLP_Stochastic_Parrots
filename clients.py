import ollama
import os
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold
from datasets import load_dataset
from typing import List, Callable
import traceback


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


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

    def query(self, prompt: str) -> str:
        if self.model_name in {"llama3.2", "deepseek-r1:8b"}:
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
    
    def _query_claude(self, prompt: str) -> str:
        raise NotImplementedError


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
