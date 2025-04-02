from abc import ABC, abstractmethod
import pandas as pd
import requests
import json
import os
from tqdm import tqdm
import time
import subprocess
import sys
import tempfile
import shutil
import re
import random
from pathlib import Path
import git

# Base Agent class
class Agent(ABC):
    def __init__(self, api_key, model_name="gemini-2.5-pro-exp-03-25"):
        self.api_key = api_key
        self.model_name = model_name
        
    def query_gemini_direct(prompt, max_tokens=65536, max_retries=3):
        """
        Query Gemini model directly via Google's API
        
        Parameters:
        prompt (str): The prompt to send to the model
        max_tokens (int): Maximum number of tokens in the response
        max_retries (int): Maximum number of retries for rate limiting
        
        Returns:
        str: The model's response
        """
        # First, ensure the google-generativeai package is installed
        try:
            import google.generativeai as genai
        except ImportError:
            print("Installing google-generativeai package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
            # Import again after installation
            import google.generativeai as genai

        # Initialize the Gemini client
        genai.configure(api_key=self.api_key)
        
        # Configure generation parameters
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": 0.2,  # Lower temperature for code generation tasks
            "top_p": 0.95,
            "top_k": 40,
        }
        
        # Implement exponential backoff for rate limiting
        for attempt in range(max_retries):
            try:
                # Generate content
                model = genai.GenerativeModel(model_name=self.model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                    ]
                )
                
                # Handle empty responses
                if not hasattr(response, 'candidates') or not response.candidates:
                    print("Warning: Empty response received from Gemini API")
                    return "The model returned an empty response. Please try again with a different prompt."
                
                # Extract the text from all candidates to ensure we get the full response
                full_text = ""
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    full_text += part.text
                
                # If we got text, return it
                if full_text:
                    return full_text
                    
                # If we couldn't extract text in the usual way, try the response.text property
                # but handle the case where it might raise an exception
                try:
                    return response.text
                except Exception as text_error:
                    print(f"Warning: Could not extract text from response: {text_error}")
                    # Try to extract any text we can find in the response object
                    if hasattr(response, '__dict__'):
                        return f"Error extracting response text. Raw response: {str(response.__dict__)}"
                    return "Error extracting response text. Please try again."
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limiting error
                if "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                    # Calculate retry delay with exponential backoff
                    retry_after = (2 ** attempt) + random.uniform(0, 1)
                    
                    # Try to extract a specific retry time if available
                    retry_match = re.search(r'retry\s+in\s+(\d+)', error_str, re.IGNORECASE)
                    if retry_match:
                        try:
                            retry_after = int(retry_match.group(1))
                        except:
                            pass  # Use the calculated value if conversion fails
                    
                    print(f"Rate limit exceeded, retrying in {retry_after} seconds (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_after)
                    continue
                else:
                    # For other errors, print the error and try again with a different approach
                    print(f"Error calling Gemini API: {e}")
                    
                    # If we've tried enough times with the current approach, try a different one
                    if attempt < max_retries - 1:
                        # Try with a simpler configuration on the next attempt
                        if attempt == max_retries - 2:
                            print("Trying with simpler configuration...")
                            generation_config = {
                                "max_output_tokens": max_tokens,
                                "temperature": 0.0,
                            }
                        time.sleep(2)  # Brief pause before retry
                        continue
                    else:
                        raise
        
        # If we've exhausted all retries
        raise Exception(f"Failed to query Gemini API after {max_retries} attempts due to rate limiting")
    
    @abstractmethod
    def forward(self, data):
        """Process input data according to agent's role"""
        pass

# RAG Analyzer Agent
class RAGAnalyser(Agent):
    def __init__(self, api_key, model_name="gemini-2.5-pro-exp-03-25"):
        super().__init__(api_key, model_name)
        self.knowledge_base = {}
        
        # Load dataset once during initialization
        from datasets import load_dataset
        print("Loading dataset...")
        self.ds = load_dataset("bigscience-data/roots_code_stackexchange")
        print(f"Dataset loaded with {len(self.ds['train'])} examples")
        
        # Initialize encoder once
        from sentence_transformers import SentenceTransformer
        model_name = "all-MiniLM-L6-v2"  # Lightweight model for embeddings
        self.encoder = SentenceTransformer(model_name)
        print(f"Loaded encoder model: {model_name}")
    
    def forward(self, test_case, num_rags=10):
        """
        Find the most similar problems to the given test case issue using precomputed embeddings
        
        Args:
            test_case (dict): Test case containing issue description
            
        Returns:
            list: List of texts of the most similar problems
        """
        import numpy as np
        import pandas as pd
        import torch
        import os
        import glob
        from sklearn.metrics.pairwise import cosine_similarity
        
        RAGS = []
        
        # Extract issue description from test case
        issue_description = test_case.get('issue_description', '')
        
        if not issue_description:
            print("Warning: No issue description provided")
            return RAGS
        
        try:
            # Encode issue description
            print("Encoding issue description...")
            query_embedding = self.encoder.encode(issue_description, convert_to_tensor=True)
            query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
            
            # Initialize structure to track top matches
            top_n = 10
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
            
        except Exception as e:
            print(f"Error in RAGAnalyser.forward: {e}")

        # TODO - ask gemini to select the most relevant RAGs
        
        return RAGS

# File Explorer Agent
class FileExplorer(Agent):
    def __init__(self, api_key, model_name="gemini-2.5-pro-exp-03-25"):
        super().__init__(api_key, model_name)
    
    def forward(self, test_case):
        """
        Given a test case, this function will extract the relevant files from the codebase
        and return them in a structured format: 
        {'filepath': 'path/to/file', 'content': 'file content'}
        """

        prompt = f"""
            You are a skilled software developer tasked with fixing an issue in a codebase by finding the relevant files.
            Your task is only to identify the files that are relevant to the issue, not to fix it.

            Project: {test_case.get('project', 'Unknown')}
            Issue URL: {test_case.get('issue_url', 'Unknown')}

            Issue Description:
            {test_case.get('issue_description', 'No description provided')}

            """
        
        # Add relevant files with content
        files = test_case.get('files', {})
        files_array = []
        if files:
            prompt += "\nThe following files are relevant to the issue:\n"
            for filepath, content in files.items():
                prompt += f"\nFile: {filepath}\n```\n{content}\n```\n"
                files_array.append({'filepath': filepath, 'content': content})
        else:
            # Fallback for cases where file extraction didn't work properly
            # Include the raw patches
            prompt += "\nThe following patches show the changes that need to be made:\n"
            if 'patch' in test_case:
                prompt += f"\nMain code patch:\n```diff\n{test_case.get('patch', '')}\n```\n"
            if 'test_patch' in test_case:
                prompt += f"\nTest code patch:\n```diff\n{test_case.get('test_patch', '')}\n```\n"
        
        # Add some context about tests if available
        if 'FAIL_TO_PASS' in test_case:
            fail_to_pass = test_case.get('FAIL_TO_PASS', '[]')
            if isinstance(fail_to_pass, str):
                try:
                    fail_to_pass = json.loads(fail_to_pass)
                except:
                    pass
            if fail_to_pass:
                prompt += f"\nThe following tests fail to pass because of the issue: {fail_to_pass}\n"
        
        prompt += """
                Please provide a list of files that are relevant to the issue (do not write the code to solve it) in this format:
                [filename1, filename2, ...]
                If you are unable to determine the relevant files, please respond with [].
                List only the filepaths, without any additional text or explanations.
                """

        response = self.query_gemini_direct(prompt)

        relevant_files_from_gemini = []
        try:
            response = response.split('[')[1].split(']')[0]
            relevant_files_from_gemini = [filename.strip().strip('"') for filename in response.split(',')]
        except:
            return []
        
        relevant_files = []
        for filename in relevant_files_from_gemini:
            for file in files_array:
                if filename in file['filepath']:
                    relevant_files.append(file)

        # returns a dictionary in the format:
        # {'filepath': 'path/to/file', 'content': 'file content'}
        return relevant_files
    

# Programmer Agent
class Programmer(Agent):
    def __init__(self, api_key, model_name="gemini-2.5-pro-exp-03-25"):
        super().__init__(api_key, model_name)
    
    def forward(self, test_case, RAGS, relevant_files):
        # test case is 


        statuses_to_return = ["OK", "FAIL", "ERROR"]
        # example
        return {"status": statuses_to_return[0]}
    

def run_agents(test_case, max_iterations, api_key, model_name="gemini-2.5-pro-exp-03-25"):
    # Initialize agents
    file_explorer = FileExplorer(api_key, model_name)
    rag_analyser = RAGAnalyser(api_key, model_name)
    programmer = Programmer(api_key, model_name)

    code_changes = {"status": "FAIL"}

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")

        # Run file explorer agent
        relevant_files = file_explorer.forward(test_case)
        print(f"Relevant files: {relevant_files}")

        # Run RAG analyser agent
        rags = rag_analyser.forward(test_case, num_rags=10)
        print(f"RAGS: {rags}")

        # Run programmer agent
        code_changes = programmer.forward(test_case, rags, relevant_files)
        print(f"Status: {code_changes['status']}")

        if code_changes['status'] == "OK":
            print("Code changes applied successfully.")
            break

        # change the test case to reflect the new errors
        # idk how to design this yet
        # depends on what Hadi's programmer agent returns
    
    return code_changes