import glob
import os
import traceback
from typing import Any, Callable, List, Optional, Type, no_type_check

import dotenv
import google.generativeai as genai
import numpy as np
import ollama
import pandas as pd
from datasets import DatasetDict, load_dataset
from google.generativeai.types import GenerateContentResponse
from google.generativeai.types.safety_types import (
    HarmBlockThreshold,
    HarmCategory,
)
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .envrc file
# put your API keys here
dotenv.load_dotenv(".envrc")


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


OPENROUTER_MODELS = {
    "deepseek/deepseek-r1-zero:free",
    "deepseek/deepseek-r1-distill-llama-8b",
    "deepseek/deepseek-r1-distill-qwen-1.5b",
    "deepseek/deepseek-r1-distill-qwen-32b",
    "deepseek/deepseek-r1-distill-qwen-14b",
    "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-r1",
    "anthropic/claude-3.7-sonnet",
    "x-ai/grok-3-beta",
}

GEMINI_MODELS = {"gemini-2.5-pro-exp-03-25", "gemini-1.5-pro"}
OLLAMA_MODELS = {"llama3.2", "deepseek-r1:8b"}


class Retries:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def _func_with_retries[R](
        self, func: Callable[..., R], *args, **kwargs
    ) -> R:
        for num_retries in range(self.max_retries):
            try:
                rval = func(*args, **kwargs)
                if num_retries > 0:
                    print(
                        f"Function {func} succeeded after {num_retries + 1} "
                        f"attempts"
                    )
                return rval
            except Exception:
                print(
                    f"Failed attempt {num_retries + 1} of {self.max_retries} "
                    f"running {func}"
                )
                if num_retries < self.max_retries - 1:
                    print("Retrying due to error:")
                    print(traceback.format_exc())
                else:
                    print("Max retries reached. Error:")
                    print(traceback.format_exc())
        raise RuntimeError(f"Function {func} failed after maximum retries")


class ModelClient(Retries):
    def __init__(self, model_name: str, max_retries: int = 3, temp: Optional[float] = None):
        super().__init__(max_retries=max_retries)
        self.model_name = model_name
        self.client: OpenAI | None = None
        self._request_limit_per_minute: float | None = None
        self.is_openrouter = False
        self.temp = temp

        if self.model_name in OPENROUTER_MODELS:
            self.is_openrouter = True
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )

    def query(
        self,
        prompt: str,
        structure: Optional[Any] = None
    ) -> str:
        """
        Query the model

        args
        - prompt:
            The text prompt to supply to the model
        - structure=None:
            The structure to provide the LLM for structured output.
            This must be a pydantic BaseModel for a JSON scehma - refer to the
            docs here:

            https://ai.google.dev/gemini-api/docs/structured-output?lang=python
        """
        if self.temp is not None and self.model_name not in GEMINI_MODELS:
            raise NotImplementedError(
                "Temperature not supported for this model")
        if self.is_openrouter:
            return self._func_with_retries(
                self._query_openrouter, prompt, structure=structure
            )
        elif self.model_name in OLLAMA_MODELS:
            return self._func_with_retries(
                self._query_local_ollama, prompt, structure=structure
            )
        elif self.model_name in GEMINI_MODELS:
            return self._func_with_retries(
                self._query_gemini,
                prompt,
                structure=structure,
                temp=0.7 if self.temp is None else self.temp
            )
        else:
            raise ValueError(f"Model {self.model_name} not supported")

    def _query_local_ollama(
        self, prompt: str, structure: Optional[Type[BaseModel]] = None
    ) -> str:
        """
        Queries a local ollama model

        args: read ModelClient.query()
        """
        client = ollama.Client()

        response = client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            format=structure.model_json_schema() if structure else None,
        )

        response_content = ""
        for chunk in response:
            content = chunk.message.content
            if content is None:
                raise ValueError(
                    f"Error: Received None content from model "
                    f"{self.model_name}"
                )
            response_content += content
            if len(response_content) > 50000:
                raise ValueError(
                    f"Error: Response from model {self.model_name} exceeded "
                    "50,000 characters. Likely model is repeating itself."
                )

        return response_content

    def _query_gemini(
        self,
        prompt: str,
        structure: Optional[Type[BaseModel]] = None,
        temp: float = 0.7,
    ) -> str:
        """
        Queries the Gemini API model

        args: read ModelClient.query()
        """
        genai.configure(api_key=GOOGLE_API_KEY)

        if structure is not None:
            generation_config = genai.GenerationConfig(
                temperature=temp,
                top_p=0.9,
                top_k=40,
                max_output_tokens=65536,
                response_mime_type="application/json",
                response_schema=structure,
            )
        else:
            generation_config = genai.GenerationConfig(
                temperature=temp,
                top_p=0.9,
                top_k=40,
                max_output_tokens=65536,
            )
        BLOCK_NONE = HarmBlockThreshold.BLOCK_NONE
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: BLOCK_NONE,
        }

        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        response: GenerateContentResponse = model.generate_content(prompt)
        response_content = response.text

        return response_content

    @no_type_check  # Apologies
    def _query_openrouter(
        self, prompt: str, structure: Optional[Type[BaseModel]] = None
    ) -> str:
        assert isinstance(self.client, OpenAI)

        # Query parameters
        extra_body = {
            "provider": {"sort": "throughput"},
            # "transforms": ["middle-out"],
        }
        # extra_body = None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        if structure is not None:
            structure_json = structure.model_json_schema()
            properties = {}

            for p_name, p in structure_json["properties"].items():
                properties[p_name] = {
                    "description": p.get(
                        "description", "<no description given>"
                    ),
                    "type": p["type"],
                }

            completion = self.client.chat.completions.create(
                extra_body=extra_body,
                model=self.model_name,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": structure_json["title"],
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": properties,
                            "required": list(properties),
                            "additionalProperties": False,
                        },
                    },
                },
                n=1,  # number of choices to generate
            )
        else:
            completion = self.client.chat.completions.create(
                extra_body=extra_body,
                model=self.model_name,
                messages=messages,
                n=1,  # number of choices to generate
            )

        if completion.choices is None:
            if hasattr(completion, "error"):
                raise ValueError(
                    f"Error: {completion.error["message"]} from model "
                    f"{self.model_name}"
                )
            else:
                raise ValueError(
                    f"Error: No choices returned from model {self.model_name}")

        assert len(completion.choices) == 1

        message = completion.choices[0].message

        if message.content is None:
            raise ValueError(
                f"Error: Received None content from model {self.model_name}"
            )

        return message.content


class RagClient(Retries):
    def __init__(self, max_retries: int = 3):
        super().__init__(max_retries=max_retries)
        dataset = load_dataset("bigscience-data/roots_code_stackexchange")
        assert isinstance(dataset, DatasetDict)
        self.ds: DatasetDict = dataset
        model_name = "all-MiniLM-L6-v2"
        self.encoder = SentenceTransformer(model_name)

    def query(
        self, issue_description: str, num_retrieve: int = 10
    ) -> List[str]:
        """
        Find the most similar problems to the given issue using precomputed
        embeddings.
        """
        RAGS: list[str] = []

        if not issue_description:
            print("Warning: No issue description provided")
            return RAGS

        # Encode issue description
        query_embedding = self.encoder.encode(
            issue_description, convert_to_tensor=True
        )
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)

        # Initialize structure to track top matches
        top_n = num_retrieve
        top_similarities: list[float] = []
        top_indices: list[int] = []

        # Find all parquet files with embeddings
        output_dir = "./embeddings/code_stack_exchange"  # Default to current directory, adjust as needed
        parquet_files = glob.glob(f"{output_dir}/batch_*.parquet")

        if not parquet_files:
            print(f"No precomputed embedding files found in {output_dir}")
            return RAGS

        # Process each batch file to find most similar vectors
        for batch_file in parquet_files:
            # Load batch of precomputed embeddings
            df_batch = pd.read_parquet(batch_file)

            # Process embeddings and calculate similarity
            batch_similarities_list: list[float] = []
            batch_indices_list: list[int] = []

            for _, row in df_batch.iterrows():
                idx = row["index"]
                embedding = np.array(row["embedding"]).reshape(1, -1)

                # Calculate similarity
                similarity = cosine_similarity(query_embedding_np, embedding)[
                    0
                ][0]

                batch_similarities_list.append(similarity)
                batch_indices_list.append(idx)

            # Convert to numpy arrays for efficient operations
            batch_similarities = np.array(batch_similarities_list)
            batch_indices = np.array(batch_indices_list)

            # If we don't have enough matches yet, add all from this batch
            if len(top_indices) < top_n:
                # Get all similarities from current batch
                current_top_indices = list(batch_indices)
                current_top_similarities = list(batch_similarities)

                # Combine with existing matches
                all_indices = top_indices + current_top_indices
                all_similarities = top_similarities + current_top_similarities

                # Get top N from combined results
                combined = list(zip(all_similarities, all_indices))
                combined.sort(reverse=True)

                # Keep only top N
                combined = combined[:top_n]

                # Update top lists
                top_similarities = [x[0] for x in combined]
                top_indices = [x[1] for x in combined]
            else:
                # For each embedding in batch, check if it belongs in top N
                for i in range(len(batch_indices)):
                    # If this similarity is higher than the lowest in our top
                    # list
                    if batch_similarities[i] > min(top_similarities):
                        # Add to our lists
                        top_similarities.append(batch_similarities[i])
                        top_indices.append(batch_indices[i])

                        # Re-sort and trim
                        combined = list(zip(top_similarities, top_indices))
                        combined.sort(reverse=True)
                        combined = combined[:top_n]

                        # Update top lists
                        top_similarities = [x[0] for x in combined]
                        top_indices = [x[1] for x in combined]

        # Retrieve full texts for top matches
        for i, (idx, similarity) in enumerate(
            zip(top_indices, top_similarities)
        ):
            idx = int(idx)  # Ensure index is integer
            text = self.ds["train"][idx]["text"]
            RAGS.append(text)

        return RAGS
