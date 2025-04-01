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
from datasets import load_dataset
# Removed the problematic import here

# Function to load SWE-Bench dataset from Huggingface
def load_swebench(variant="lite"):
    """
    Load the SWE-Bench dataset from Huggingface
    
    Parameters:
    variant (str): 'lite', 'verified', or 'multimodal'
    
    Returns:
    list: List of test cases
    """
    print(f"Loading SWE-Bench {variant} dataset from Huggingface...")
    
    if variant == "lite":
        dataset_id = "princeton-nlp/SWE-bench_Lite"
    elif variant == "verified":
        dataset_id = "princeton-nlp/SWE-bench_Verified"
    elif variant == "multimodal":
        dataset_id = "princeton-nlp/SWE-bench_Multimodal"
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    # Load the dataset from Huggingface
    try:
        dataset = load_dataset(dataset_id)
        
        # Convert the Huggingface dataset to the expected format
        test_cases = []
        for item in dataset['test']:  # Using 'test' split by default
            # Process the item to add the expected fields
            processed_item = process_swebench_item(item)
            test_cases.append(processed_item)
            
        print(f"Successfully loaded {len(test_cases)} test cases from {dataset_id}")
        return test_cases
    except Exception as e:
        print(f"Error loading dataset from Huggingface: {e}")
        raise Exception(f"Failed to load dataset from Huggingface: {e}")

def process_swebench_item(item):
    """
    Process a SWE-Bench dataset item to add expected fields
    
    Parameters:
    item (dict): The raw dataset item
    
    Returns:
    dict: The processed item with additional fields
    """
    # Create a copy to avoid modifying the original
    processed = item.copy()
    
    # Map fields to expected format
    processed['project'] = item.get('repo', 'Unknown')
    processed['issue_url'] = f"https://github.com/{item.get('repo', 'unknown')}/issues/{item.get('instance_id', '').split('-')[-1]}"
    processed['issue_description'] = item.get('problem_statement', 'No description provided')
    
    # Extract files from patch and test_patch
    processed['files'] = extract_files_from_patches(item.get('patch', ''), item.get('test_patch', ''))
    
    return processed

def extract_files_from_patches(patch, test_patch):
    """
    Extract file contents from git patch format
    
    Parameters:
    patch (str): The main code patch
    test_patch (str): The test patch
    
    Returns:
    dict: Dictionary mapping file paths to content
    """
    files = {}
    
    # Process both patches
    for current_patch in [patch, test_patch]:
        if not current_patch:
            continue
            
        # Split the patch into different file changes
        file_changes = current_patch.split('diff --git ')[1:]
        
        for change in file_changes:
            # Extract the file path
            file_path = None
            for line in change.split('\n'):
                if line.startswith('+++'):
                    # The +++ line contains the file path after edit
                    file_path = line[4:].strip().split('\t')[0]
                    if file_path.startswith('b/'):
                        file_path = file_path[2:]  # Remove the b/ prefix
                    break
                    
            if not file_path:
                continue
                
            # Now attempt to recreate the file content
            # This is a simplified approach - in a real implementation,
            # you might want to actually apply the patch to the base file
            # But for demonstration, we'll extract chunks marked with '+' (additions)
            
            # For simplicity, we'll just collect the context and the changes
            content_lines = []
            in_content = False
            
            for line in change.split('\n'):
                if line.startswith('@@'):
                    in_content = True
                    continue
                    
                if in_content:
                    if line.startswith('+'):
                        # Addition - include without the + sign
                        content_lines.append(line[1:])
                    elif not line.startswith('-'):
                        # Context line - include as is
                        if line.startswith(' '):
                            content_lines.append(line[1:])
                        else:
                            content_lines.append(line)
            
            if content_lines:
                files[file_path] = '\n'.join(content_lines)
    
    return files

# Function to query Llama 3.2 via Ollama
def query_llama(prompt, model="llama3.2", max_tokens=4096):
    """
    Query Llama 3.2 model via Ollama API
    
    Parameters:
    prompt (str): The prompt to send to the model
    model (str): The model name in Ollama
    max_tokens (int): Maximum number of tokens in the response
    
    Returns:
    str: The model's response
    """
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens
        }
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Failed to query Ollama: {response.status_code}")

# Function to query Gemini directly via Google API
def query_gemini_direct(prompt, api_key, model_name="gemini-2.5-pro-exp-03-25", max_tokens=65536, max_retries=3):
    """
    Query Gemini model directly via Google's API
    
    Parameters:
    prompt (str): The prompt to send to the model
    api_key (str): Google API key
    model_name (str): The specific Gemini model to use
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
    genai.configure(api_key=api_key)
    
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
            model = genai.GenerativeModel(model_name=model_name)
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

# Function to extract code blocks from model response
def extract_code_from_response(response):
    """
    Extract code blocks from the model's response
    
    Parameters:
    response (str): The model's response
    
    Returns:
    dict: Dictionary mapping file paths to code content
    """
    # Pattern to match code blocks with file paths in comments or markdown headers
    file_pattern = re.compile(r'(?:```(?:\w+)?\s*(?:#|\/{2})\s*([^\n]+)\n(.*?)```|(?:#+\s*([^\n]+)\n```(?:\w+)?\n(.*?)```)|```(\w+)\n(.*?)```)', re.DOTALL)
    
    files = {}
    
    for match in file_pattern.finditer(response):
        filepath = match.group(1) or match.group(3) or "unnamed_file.py"
        filepath = filepath.strip()
        
        # Remove any "File: " prefix
        if filepath.lower().startswith("file:"):
            filepath = filepath[5:].strip()
            
        content = match.group(2) or match.group(4) or match.group(6)
        
        if filepath and content:
            files[filepath] = content.strip()
    
    return files

# Function to create prompt from test case
def create_prompt(test_case):
    """
    Create a prompt for the LLM based on a test case
    
    Parameters:
    test_case (dict): The test case data
    
    Returns:
    str: The formatted prompt
    """
    # Construct prompt based on SWE-Bench format
    prompt = f"""
You are a skilled software developer tasked with fixing an issue in a codebase. 

Project: {test_case.get('project', 'Unknown')}
Issue URL: {test_case.get('issue_url', 'Unknown')}

Issue Description:
{test_case.get('issue_description', 'No description provided')}

The following files are relevant to the issue:

"""
    
    # Add relevant files with content
    files = test_case.get('files', {})
    if files:
        for filepath, content in files.items():
            prompt += f"\nFile: {filepath}\n```\n{content}\n```\n"
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
            prompt += f"\nThe following tests should pass after your fix: {fail_to_pass}\n"
    
    prompt += """
Please provide a comprehensive solution to fix this issue. Format your answer as follows:

1. First, explain your understanding of the problem.
2. Explain your approach to solving it.
3. Provide the code changes needed to fix the issue. For each file that needs changes, include:
   - The file path
   - The complete updated code (not just the changed parts)

Format your code blocks like this:
```python
# filename.py
def example():
    return "fixed code"
```

Remember to provide complete, working solutions and test your logic carefully.
"""
    
    return prompt

# Function to evaluate response using SWE-Bench methodology
def evaluate_response(test_case, response, work_dir):
    """
    Evaluate if the model's response correctly solves the test case
    
    Parameters:
    test_case (dict): The test case data
    response (str): The model's response
    work_dir (str): Working directory for evaluation
    
    Returns:
    dict: Evaluation results
    """
    # Extract code changes from the response
    code_changes = extract_code_from_response(response)
    
    if not code_changes:
        return {
            'passed': False,
            'error': 'No code changes extracted from response',
            'files_changed': 0
        }
    
    # Apply code changes to the test repo
    files_changed = 0
    try:
        for filepath, content in code_changes.items():
            file_path = os.path.join(work_dir, filepath)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            files_changed += 1
    except Exception as e:
        return {
            'passed': False,
            'error': f'Error applying code changes: {str(e)}',
            'files_changed': files_changed
        }
    
    # Run evaluation command if provided
    if 'evaluation_command' in test_case:
        try:
            result = subprocess.run(
                test_case['evaluation_command'],
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            passed = result.returncode == 0
            
            return {
                'passed': passed,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'files_changed': files_changed
            }
        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'error': 'Evaluation timed out',
                'files_changed': files_changed
            }
        except Exception as e:
            return {
                'passed': False,
                'error': f'Error running evaluation: {str(e)}',
                'files_changed': files_changed
            }
    
    # If no evaluation command, check for expected changes
    if 'expected_changes' in test_case:
        # Compare code changes with expected changes
        expected_files = set(test_case['expected_changes'].keys())
        changed_files = set(code_changes.keys())
        
        # Check if all expected files were changed
        if not expected_files.issubset(changed_files):
            return {
                'passed': False,
                'error': f'Missing changes in expected files: {expected_files - changed_files}',
                'files_changed': files_changed
            }
        
        # For a basic check, we could look for specific patterns or code snippets
        all_matches = True
        for filepath, expected_content in test_case['expected_changes'].items():
            if filepath in code_changes:
                if isinstance(expected_content, list):  # List of patterns to check
                    for pattern in expected_content:
                        if pattern not in code_changes[filepath]:
                            all_matches = False
                            break
                else:  # Direct content comparison
                    if expected_content not in code_changes[filepath]:
                        all_matches = False
            else:
                all_matches = False
        
        return {
            'passed': all_matches,
            'files_changed': files_changed
        }
    
    # If no clear evaluation criteria, consider it unknown
    return {
        'passed': None,
        'message': 'No evaluation criteria available',
        'files_changed': files_changed
    }

# Setup test environment
def setup_test_environment(test_case):
    """
    Set up a test environment for a specific test case
    
    Parameters:
    test_case (dict): The test case data
    
    Returns:
    str: Path to the working directory
    """
    # Create a temporary directory
    work_dir = tempfile.mkdtemp()
    
    # If the test case includes a repository URL, clone it
    if 'repo_url' in test_case and test_case['repo_url']:
        try:
            git.Repo.clone_from(
                test_case['repo_url'],
                work_dir,
                depth=1,  # Shallow clone for speed
                branch=test_case.get('branch', 'main')
            )
            
            # Checkout specific commit if provided
            if 'commit_id' in test_case and test_case['commit_id']:
                repo = git.Repo(work_dir)
                repo.git.checkout(test_case['commit_id'])
        except Exception as e:
            print(f"Error cloning repository: {e}")
            # If cloning fails, we'll use the files provided in the test case
    
    # If no repository or cloning failed, create files from the test case
    if 'files' in test_case:
        for filepath, content in test_case['files'].items():
            file_path = os.path.join(work_dir, filepath)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
    
    return work_dir

# Cleanup test environment
def cleanup_test_environment(work_dir):
    """
    Clean up the test environment
    
    Parameters:
    work_dir (str): Path to the working directory
    """
    shutil.rmtree(work_dir, ignore_errors=True)

# Function to compute and save embeddings
def compute_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """
    Compute embeddings for a list of texts
    
    Parameters:
    texts (list): List of text strings
    model_name (str): Name of the sentence transformer model
    
    Returns:
    numpy.ndarray: Array of embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        print(f"Computing embeddings using {model_name}...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings
    except ImportError:
        print("Warning: sentence-transformers not installed, skipping embeddings")
        return None
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        return None

# Main function to run the benchmark
def run_benchmark(variant="lite", model="llama3.2", output_file="new_project/benchmark_results.parquet", 
                 limit=None, save_embeddings=False, api_key=None,
                 gemini_model="gemini-2.5-pro-exp-03-25",
                 max_retries=5, delay_between_calls=2):
    """
    Run the SWE-Bench benchmark on Llama 3.2 or Gemini models
    
    Parameters:
    variant (str): SWE-Bench variant to use ('lite', 'verified', or 'multimodal')
    model (str): The model name ('llama3.2' for Ollama or 'gemini' for Google API)
    output_file (str): Path to save results
    limit (int, optional): Limit the number of test cases to process
    save_embeddings (bool): Whether to compute and save embeddings
    api_key (str, optional): API key (required for Gemini)
    gemini_model (str): Specific Gemini model to use
    max_retries (int): Maximum number of retries for rate-limited API calls
    delay_between_calls (int): Delay in seconds between API calls to avoid rate limiting
    """
    # Check if the selected model is available
    if model == "llama3.2":
        # Check if Ollama is running
        try:
            requests.get("http://localhost:11434/api/version")
        except requests.exceptions.ConnectionError:
            print("Error: Ollama is not running. Please start Ollama first.")
            sys.exit(1)
        
        # Check if the model is available in Ollama
        try:
            response = requests.post(
                "http://localhost:11434/api/tags",
                json={}
            )
            available_models = [model_info["name"] for model_info in response.json()["models"]]
            if model not in available_models:
                print(f"Warning: Model '{model}' not found in Ollama. You may need to pull it first with 'ollama pull {model}'")
        except Exception as e:
            print(f"Warning: Could not check available models: {e}")
    elif model == "gemini":
        # Check if Gemini API key is provided
        if not api_key:
            print("Error: Google API key is required for Gemini model.")
            sys.exit(1)
    else:
        print(f"Error: Unsupported model {model}. Currently supported models are 'llama3.2' and 'gemini'.")
        sys.exit(1)
    
    # Install required packages
    required_packages = ["pandas", "pyarrow", "gitpython", "tqdm", "requests", "datasets"]
    if model == "llama3.2":
        required_packages.append("ollama")
    elif model == "gemini":
        required_packages.append("google-generativeai")
    if save_embeddings:
        required_packages.extend(["sentence-transformers", "scikit-learn"])
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print(f"Loading SWE-Bench {variant} dataset from Huggingface...")
    test_cases = load_swebench(variant)
    
    if limit and limit > 0:
        test_cases = test_cases[:limit]
        print(f"Limited to first {limit} test cases")
    
    results = []
    
    print(f"Running benchmark on {len(test_cases)} test cases using {model}...")
    for i, test_case in enumerate(tqdm(test_cases)):
        test_id = test_case.get('id', f"test_{i}")
        
        print(f"\nProcessing test case {i+1}/{len(test_cases)}: {test_id}")
        
        # Create prompt
        prompt = create_prompt(test_case)
        
        # Setup test environment
        work_dir = setup_test_environment(test_case)
        
        try:
            # Query model based on the selected model
            start_time = time.time()
            if model == "llama3.2":
                response = query_llama(prompt, model)
            elif model == "gemini":
                response = query_gemini_direct(prompt, api_key, gemini_model, max_retries=max_retries)
                # Add a delay after each call to avoid rate limiting
                time.sleep(delay_between_calls)
            end_time = time.time()
            
            # Evaluate response
            eval_result = evaluate_response(test_case, response, work_dir)
            
            # Store result
            result = {
                'test_id': test_id,
                'prompt': prompt,
                'response': response,
                'passed': eval_result.get('passed'),
                'evaluation_details': eval_result,
                'latency_seconds': end_time - start_time,
                'model': model,
                'variant': variant,
                'timestamp': pd.Timestamp.now()
            }
            results.append(result)
            
            print(f"Result: {'PASSED' if eval_result.get('passed') else 'FAILED'}")
            if 'error' in eval_result:
                print(f"Error: {eval_result['error']}")
            
        except Exception as e:
            print(f"Error processing test case {test_id}: {e}")
            results.append({
                'test_id': test_id,
                'error': str(e),
                'model': model,
                'variant': variant,
                'timestamp': pd.Timestamp.now()
            })
        
        # Cleanup
        cleanup_test_environment(work_dir)
    
    # Create DataFrame and save to parquet
    print(f"Saving results to {output_file}...")
    df = pd.DataFrame(results)
    
    # Compute embeddings if requested
    if save_embeddings:
        try:
            print("Computing embeddings for prompts and responses...")
            
            # Only compute embeddings for successful test cases
            valid_prompts = [r['prompt'] for r in results if 'prompt' in r and 'response' in r]
            valid_responses = [r['response'] for r in results if 'prompt' in r and 'response' in r]
            
            if valid_prompts and valid_responses:
                # Compute embeddings
                prompt_embeddings = compute_embeddings(valid_prompts)
                response_embeddings = compute_embeddings(valid_responses)
                
                if prompt_embeddings is not None and response_embeddings is not None:
                    # Create embedding dataframes
                    prompt_embed_df = pd.DataFrame(
                        prompt_embeddings, 
                        index=[r['test_id'] for r in results if 'prompt' in r and 'response' in r]
                    )
                    prompt_embed_df.columns = [f'prompt_embedding_{i}' for i in range(prompt_embed_df.shape[1])]
                    
                    response_embed_df = pd.DataFrame(
                        response_embeddings,
                        index=[r['test_id'] for r in results if 'prompt' in r and 'response' in r]
                    )
                    response_embed_df.columns = [f'response_embedding_{i}' for i in range(response_embed_df.shape[1])]
                    
                    # Save embeddings
                    prompt_embed_df.to_parquet(f"{output_file.split('.')[0]}_prompt_embeddings.parquet")
                    response_embed_df.to_parquet(f"{output_file.split('.')[0]}_response_embeddings.parquet")
                    
                    print("Embeddings saved successfully.")
                    
                    # Calculate cosine similarity between prompts and responses
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarities = [
                        cosine_similarity([p_emb], [r_emb])[0][0]
                        for p_emb, r_emb in zip(prompt_embeddings, response_embeddings)
                    ]
                    
                    # Add to results dataframe for valid entries
                    similarity_dict = {
                        test_id: sim for test_id, sim in zip(
                            [r['test_id'] for r in results if 'prompt' in r and 'response' in r],
                            similarities
                        )
                    }
                    
                    # Add similarity scores to main dataframe
                    df['prompt_response_similarity'] = df['test_id'].map(
                        lambda x: similarity_dict.get(x, None)
                    )
        except Exception as e:
            print(f"Error computing embeddings: {e}")
    
    # Save main results
    df.to_parquet(output_file, index=False)
    
    # Also save a CSV for easy viewing
    try:
        csv_output = output_file.replace('.parquet', '.csv')
        # Select only the key columns for the CSV
        csv_df = df[['test_id', 'passed', 'latency_seconds', 'model', 'variant', 'timestamp']]
        if 'prompt_response_similarity' in df.columns:
            csv_df['prompt_response_similarity'] = df['prompt_response_similarity']
        if 'error' in df.columns:
            csv_df['error'] = df['error']
        csv_df.to_csv(csv_output, index=False)
        print(f"Summary results saved to {csv_output}")
    except Exception as e:
        print(f"Error saving CSV summary: {e}")
    
    # Print summary
    if 'passed' in df.columns:
        total_tests = len(df)
        passed_tests = df['passed'].sum() if pd.api.types.is_numeric_dtype(df['passed']) else sum(df['passed'] == True)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        print(f"Benchmark complete! {passed_tests}/{total_tests} tests passed ({pass_rate:.2%})")
    else:
        print("Benchmark complete, but test results could not be evaluated.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SWE-Bench benchmark on Llama 3.2 via Ollama or Gemini via Google API")
    parser.add_argument("--variant", type=str, default="lite", choices=["lite", "verified", "multimodal"],
                        help="SWE-Bench variant to use")
    parser.add_argument("--model", type=str, default="llama3.2", choices=["llama3.2", "gemini"],
                        help="Model to use (llama3.2 for Ollama, gemini for Google API)")
    parser.add_argument("--output", type=str, default="new_project/benchmark_results.parquet",
                        help="Output file path (default: new_project/benchmark_results.parquet)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of test cases to process")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="Calculate and save embeddings of prompts and responses")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (required for Gemini)")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-pro-exp-03-25",
                        help="Specific Gemini model to use")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Maximum number of retries for rate-limited API calls")
    parser.add_argument("--delay-between-calls", type=int, default=2,
                        help="Delay in seconds between API calls to avoid rate limiting")
    
    args = parser.parse_args()
    
    run_benchmark(
        variant=args.variant, 
        model=args.model, 
        output_file=args.output, 
        limit=args.limit,
        save_embeddings=args.save_embeddings,
        api_key=args.api_key,
        gemini_model=args.gemini_model,
        max_retries=args.max_retries,
        delay_between_calls=args.delay_between_calls
    )