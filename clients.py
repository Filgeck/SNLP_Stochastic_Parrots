from datasets import load_dataset
import anthropic
import ollama
import os
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from google.generativeai.types.safety_types import HarmCategory, HarmBlockThreshold
import ollama
import os
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Callable, List, Optional, Type
import traceback
import dotenv

# Load environment variables from .envrc file
# put your API keys here
dotenv.load_dotenv(".envrc")


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


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

    def query(self, prompt: str, structure: Optional[Type[BaseModel]] = None) -> str:
        """
        Query the model
    
        args
        - prompt: 
            The text prompt to supply to the model
        - structure=None:
            The structure to provide the LLM for structured output.
            This must be a pydantic BaseModel for a JSON scehma - refer to the docs here:

            https://ai.google.dev/gemini-api/docs/structured-output?lang=python
        """
        if self.model_name in {"llama3.2", "deepseek-r1:8b"}:
            return self._func_with_retries(self._query_local_ollama, prompt, structure=structure)
        elif self.model_name in {"gemini-2.5-pro-exp-03-25", "gemini-1.5-pro"}:
            return self._func_with_retries(self._query_gemini, prompt, structure=structure)
        elif self.model_name in {"claude-3-7-sonnet-20250219"}:
            return self._func_with_retries(self._query_claude, prompt)
        else:
            raise ValueError(f"Model {self.model_name} not supported")

    def _query_local_ollama(self, prompt: str, structure: Optional[Type[BaseModel]] = None) -> str:
        """
        Queries a local ollama model

        args: read ModelClient.query()
        """
        client = ollama.Client()

        response = client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            format=structure.model_json_schema() if structure else None
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
                    f"Error: Response from model {self.model_name} exceeded 50,000 characters. " \
                    "Likely model is repeating itself"
                )

        return response_content

    def _query_gemini(self, prompt: str, structure: Optional[Type[BaseModel]] = None) -> str:
        """
        Queries the Gemini API model

        args: read ModelClient.query()
        """
        genai.configure(api_key=GOOGLE_API_KEY)

        if structure is not None:
            generation_config = genai.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                max_output_tokens=16384,
                response_mime_type='application/json',
                response_schema=structure
            )
        else:
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

        response: GenerateContentResponse = model.generate_content(prompt)
        response_content = response.text

        return response_content
    
    def _query_claude(self, prompt: str) -> str:
        client = anthropic.Anthropic()

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
    def __init__ (self, max_retries: int = 3):
        super().__init__(max_retries=max_retries)
        self.ds = load_dataset("bigscience-data/roots_code_stackexchange")
        model_name = "all-MiniLM-L6-v2"
        self.encoder = SentenceTransformer(model_name)

    def query(self, issue_description: str, num_retrieve: int = 10) -> List[str]:
        """
        Find the most similar problems to the given issue using precomputed embeddings.
        """
        import numpy as np
        import pandas as pd
        import torch
        import os
        import glob
        from sklearn.metrics.pairwise import cosine_similarity
        
        RAGS = []
        
        if not issue_description:
            print("Warning: No issue description provided")
            return RAGS
        
        # Encode issue description
        query_embedding = self.encoder.encode(issue_description, convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        
        # Initialize structure to track top matches
        top_n = num_retrieve
        top_indices = []
        top_similarities = []
        
        # Find all parquet files with embeddings
        output_dir = "."  # Default to current directory, adjust as needed
        parquet_files = glob.glob(f"{output_dir}/batch_*.parquet")
        
        if not parquet_files:
            print(f"No precomputed embedding files found in {output_dir}")
            return RAGS
            
        print(f"Found {len(parquet_files)} embedding batch files")
        
        # Process each batch file to find most similar vectors
        for batch_file in parquet_files:
            print(f"Processing {batch_file}...")
            
            # Load batch of precomputed embeddings
            df_batch = pd.read_parquet(batch_file)
            
            # Process embeddings and calculate similarity
            batch_similarities = []
            batch_indices = []
            
            for _, row in df_batch.iterrows():
                idx = row['index']
                embedding = np.array(row['embedding']).reshape(1, -1)
                
                # Calculate similarity
                similarity = cosine_similarity(query_embedding_np, embedding)[0][0]
                
                batch_similarities.append(similarity)
                batch_indices.append(idx)
            
            # Convert to numpy arrays for efficient operations
            batch_similarities = np.array(batch_similarities)
            batch_indices = np.array(batch_indices)
            
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
                    # If this similarity is higher than the lowest in our top list
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
        print(f"Retrieved {len(top_indices)} most similar questions")
        for i, (idx, similarity) in enumerate(zip(top_indices, top_similarities)):
            idx = int(idx)  # Ensure index is integer
            text = self.ds['train'][idx]["text"]
            print(f"Match {i+1} (score: {similarity:.4f}): Original index {idx}")
            RAGS.append(text)

        return RAGS
        
