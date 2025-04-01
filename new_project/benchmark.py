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
from pathlib import Path
import git
from datasets import load_dataset

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
        for item in dataset['test']:  # Assuming 'test' is the split we want
            test_cases.append(item)
            
        print(f"Successfully loaded {len(test_cases)} test cases from {dataset_id}")
        return test_cases
    except Exception as e:
        print(f"Error loading dataset from Huggingface: {e}")
        raise Exception(f"Failed to load dataset from Huggingface: {e}")

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

# Function to query Gemini Pro 2.5 via OpenRouter API
def query_gemini_openrouter(prompt, api_key, site_url="https://example.com", site_name="Benchmark Test", max_tokens=4096):
    """
    Query Gemini Pro 2.5 Experimental model via OpenRouter API
    
    Parameters:
    prompt (str): The prompt to send to the model
    api_key (str): OpenRouter API key
    site_url (str): Site URL for rankings on openrouter.ai
    site_name (str): Site name for rankings on openrouter.ai
    max_tokens (int): Maximum number of tokens in the response
    
    Returns:
    str: The model's response
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": site_url,
        "X-Title": site_name,
    }
    
    data = {
        "model": "google/gemini-2.5-pro-exp-03-25:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": max_tokens
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        response_data = response.json()
        # Extract the text from the response
        if "choices" in response_data and len(response_data["choices"]) > 0:
            message = response_data["choices"][0].get("message", {})
            if isinstance(message, dict) and "content" in message:
                return message["content"]
        
        # Fallback in case the response structure is different
        return str(response_data)
    else:
        raise Exception(f"Failed to query OpenRouter: {response.status_code}, {response.text}")

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
    for filepath, content in test_case.get('files', {}).items():
        prompt += f"\nFile: {filepath}\n```\n{content}\n```\n"
    
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
                 limit=None, save_embeddings=False, openrouter_api_key=None,
                 site_url="https://example.com", site_name="Benchmark Test"):
    """
    Run the SWE-Bench benchmark on Llama 3.2 or Gemini Pro 2.5 Experimental
    
    Parameters:
    variant (str): SWE-Bench variant to use ('lite', 'verified', or 'multimodal')
    model (str): The model name ('llama3.2' for Ollama or 'gemini-pro-2.5' for OpenRouter)
    output_file (str): Path to save results
    limit (int, optional): Limit the number of test cases to process
    save_embeddings (bool): Whether to compute and save embeddings
    openrouter_api_key (str, optional): OpenRouter API key (required for Gemini Pro 2.5)
    site_url (str): Site URL for rankings on openrouter.ai (used with OpenRouter)
    site_name (str): Site name for rankings on openrouter.ai (used with OpenRouter)
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
    elif model == "gemini-pro-2.5":
        # Check if OpenRouter API key is provided
        if not openrouter_api_key:
            print("Error: OpenRouter API key is required for Gemini Pro 2.5 model.")
            sys.exit(1)
    else:
        print(f"Error: Unsupported model {model}. Currently supported models are 'llama3.2' and 'gemini-pro-2.5'.")
        sys.exit(1)
    
    # Install required packages
    required_packages = ["pandas", "pyarrow", "gitpython", "tqdm", "requests", "datasets"]
    if model == "llama3.2":
        required_packages.append("ollama")
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
        
        time.sleep(0.2)

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
            elif model == "gemini-pro-2.5":
                response = query_gemini_openrouter(prompt, openrouter_api_key, site_url, site_name)
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
    
    parser = argparse.ArgumentParser(description="Run SWE-Bench benchmark on Llama 3.2 via Ollama or Gemini Pro 2.5 via OpenRouter")
    parser.add_argument("--variant", type=str, default="lite", choices=["lite", "verified", "multimodal"],
                        help="SWE-Bench variant to use")
    parser.add_argument("--model", type=str, default="llama3.2", choices=["llama3.2", "gemini-pro-2.5"],
                        help="Model to use (llama3.2 for Ollama, gemini-pro-2.5 for OpenRouter)")
    parser.add_argument("--output", type=str, default="new_project/benchmark_results.parquet",
                        help="Output file path (default: new_project/benchmark_results.parquet)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of test cases to process")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="Calculate and save embeddings of prompts and responses")
    parser.add_argument("--openrouter-api-key", type=str, default=None,
                        help="OpenRouter API key (required for Gemini Pro 2.5)")
    parser.add_argument("--site-url", type=str, default="https://example.com",
                        help="Site URL for rankings on openrouter.ai (used with OpenRouter)")
    parser.add_argument("--site-name", type=str, default="Benchmark Test",
                        help="Site name for rankings on openrouter.ai (used with OpenRouter)")
    
    args = parser.parse_args()
    
    run_benchmark(
        variant=args.variant, 
        model=args.model, 
        output_file=args.output, 
        limit=args.limit,
        save_embeddings=args.save_embeddings,
        openrouter_api_key=args.openrouter_api_key,
        site_url=args.site_url,
        site_name=args.site_name
    )