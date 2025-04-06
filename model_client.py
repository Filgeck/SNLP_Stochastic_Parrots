import ollama
from ollama import ChatResponse
import os
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse, GenerationConfigType


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class ModelClient:
    def __init__(self, model_name: str, max_retries: int = 3):
        self.model_name = model_name
        self.max_retries = max_retries

    def query(self, prompt: str) -> str | None:
        if self.model_name in {"llama3.2", "deepseek-coder-v2", "gemini-1.5-pro"}:
            return self._query_with_retries(self._query_local_ollama, prompt)
        elif self.model_name == "gemini-2.5-pro-exp-03-25":
            return self._query_with_retries(self._query_gemini, prompt)
        else:
            raise ValueError(f"Model {self.model_name} not supported")

    def _query_with_retries(self, func: callable, *args, **kwargs) -> str | None:
        for num_retries in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                print(f"Failed attempt {num_retries + 1} of {self.max_retries}")
                if num_retries < self.max_retries - 1:
                    print(f"Retrying due to error: {error}")
                else:
                    print(f"Max retries reached. Error:\n{error}\n")
                    return None

    def _query_local_ollama(self, prompt: str) -> str | None:
        client = ollama.Client()

        response: ChatResponse = client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        response_content = ""
        for chunk in response:
            response_content += chunk.content
            if len(response_content) > 50000:
                print(
                    f"Error: Response from model {self.model_name} exceeded 50,000 characters. Likely model is repeating itself"
                )
                return None

        return response_content

    def _query_gemini(self, prompt: str) -> str | None:
        genai.configure(api_key=GOOGLE_API_KEY)

        generation_config: GenerationConfigType = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 16384,
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        try:
            response: GenerateContentResponse = model.generate_content(
                prompt,
            )
            response_content = response.text
        except Exception as error:
            raise error  # re-raise exception to trigger retry in _query_with_retries

        return response_content
