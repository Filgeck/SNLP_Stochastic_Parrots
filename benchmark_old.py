import os
import json
import logging
import re
from pathlib import Path
import time

import ollama
import google.generativeai as genai
from datasets import load_dataset
from tqdm.auto import tqdm
from dotenv import load_dotenv

# --- Configuration ---
LLAMA_MODEL_NAME = "llama3.2" # Adjust if your ollama model name is different (e.g., llama3.1:70b)
GEMINI_MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Using the latest stable 1.5 pro. Adjust if needed and available.
# The specific experimental model "gemini-1.5-pro-exp-03-25" might not be directly available via the standard API.
# If you have special access, use that name. Otherwise, 1.5-pro-latest is the closest generally available.

SWE_BENCH_LITE_DATASET = "princeton-nlp/SWE-bench_Lite"
SWE_BENCH_BM25_DATASET = "princeton-nlp/SWE-bench_bm25_40K" # Contains the 'text' field
OUTPUT_FILE = Path("all_pred.jsonl")
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load API Key ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Helper Functions ---

def build_prompt(text_context: str) -> str:
    """Builds the prompt for the LLM."""
    # The 'text' field already contains the issue and code context structure
    # We just need to add the instruction for the patch format
    instruction = (
        """Feel free to analyse and edit files as required, however you must absolutely ensure that at the end of your response you enclose your final patch in <patch> </patch> tags."""
    )
    return f"{text_context}\n\n{instruction}"

def extract_patch(response_text: str) -> str:
    """Extracts the patch content from the LLM response."""
    # Use regex to find the content between <patch> and </patch> tags
    # re.DOTALL makes '.' match newline characters as well
    match = re.search(r"<patch>\s*\n?(.*?)\n?\s*</patch>", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        # Strip leading/trailing whitespace from the extracted patch content
        return match.group(1) # .strip()
    else:
        logging.warning("Could not find <patch>...</patch> tags in response. Returning raw response.")
        # Fallback: return the raw response, maybe it's just the diff directly
        # Or return empty string if it's likely just conversational text
        # Let's return the stripped raw response as a best guess fallback
        return response_text.strip()

def get_ollama_patch(client, model_name: str, prompt: str) -> str | None:
    """Gets a patch prediction from a local Ollama model."""
    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"Attempt {attempt+1}/{MAX_RETRIES} for Ollama model {model_name}")
            response = client.generate(
                model=model_name,
                prompt=prompt,
                # Options can be added here if needed, e.g., temperature
                # options={'temperature': 0.0}
            )
            # Check if response content exists
            if 'response' in response and response['response']:
                raw_text = response['response']
                # if not raw_text.strip().startswith("<patch>") or not raw_text.strip().endswith("</patch>"):
                #     raw_text = f"<patch>\n{raw_text.strip()}\n</patch>"
                return extract_patch(raw_text)
            else:
                logging.warning(f"Ollama response missing 'response' content: {response}")
                return "" # Return empty string if no valid response content
        except Exception as e:
            logging.error(f"Error getting patch from Ollama {model_name} (Attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                logging.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error(f"Max retries reached for Ollama model {model_name}.")
                return None # Indicate failure after retries
    return None


def get_gemini_patch(model, prompt: str) -> str | None:
    """Gets a patch prediction from the Gemini API."""
    # Configure safety settings to be less restrictive, potentially needed for code/diffs
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    # Generation config (optional, e.g., temperature=0.0 for deterministic output)
    generation_config = genai.types.GenerationConfig(
        # candidate_count=1, # Default is 1
        # stop_sequences=["</patch>"], # Can sometimes help, but also truncate valid output
        temperature=1.0 # Make output more deterministic
    )

    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"Attempt {attempt+1}/{MAX_RETRIES} for Gemini model {model.model_name}")
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            # Accessing the text might differ slightly based on API version/response structure
            # Use response.text for simplicity, check response.parts if needed
            if response.text:
                 raw_text = response.text
                 return extract_patch(raw_text)
            elif response.parts:
                 # Combine text from parts if .text is empty
                 raw_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                 if raw_text:
                     return extract_patch(raw_text)
                 else:
                    logging.warning(f"Gemini response has parts but no text content: {response.parts}")
                    # Check for finish_reason (e.g., safety)
                    try:
                        logging.warning(f"Gemini finish reason: {response.candidates[0].finish_reason}")
                        logging.warning(f"Gemini safety ratings: {response.candidates[0].safety_ratings}")
                    except (AttributeError, IndexError):
                        pass # Ignore if finish_reason/safety_ratings can't be accessed
                    return ""
            else:
                logging.warning(f"Gemini response seems empty or invalid: {response}")
                # Log finish reason if available
                try:
                    logging.warning(f"Gemini finish reason: {response.candidates[0].finish_reason}")
                    logging.warning(f"Gemini safety ratings: {response.candidates[0].safety_ratings}")
                except (AttributeError, IndexError):
                     pass # Ignore if finish_reason/safety_ratings can't be accessed
                return "" # Return empty string for empty/invalid responses

        except Exception as e:
            logging.error(f"Error getting patch from Gemini {model.model_name} (Attempt {attempt+1}): {e}")
            # Specific check for common API errors like ResourceExhausted (rate limits)
            if "Resource has been exhausted" in str(e) or "429" in str(e):
                 wait_time = RETRY_DELAY_SECONDS * (attempt + 1) # Exponential backoff might be better
                 logging.warning(f"Rate limit likely hit. Waiting {wait_time} seconds.")
                 time.sleep(wait_time)
            elif attempt < MAX_RETRIES - 1:
                logging.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                 logging.error(f"Max retries reached for Gemini model {model.model_name}.")
                 return None # Indicate failure after retries
    return None


# --- Main Script ---
if __name__ == "__main__":
    logging.info("Starting SWE-bench Lite baseline generation...")

    # --- Load Datasets ---
    logging.info("Loading SWE-bench Lite dataset...")
    try:
        swe_lite = load_dataset(SWE_BENCH_LITE_DATASET)
        swe_lite_test = swe_lite['test']
        logging.info(f"Loaded {len(swe_lite_test)} test instances from SWE-bench Lite.")
    except Exception as e:
        logging.error(f"Failed to load SWE-bench Lite dataset: {e}")
        exit(1)

    logging.info("Loading SWE-bench BM25 dataset for context...")
    try:
        # Use streaming=True if the dataset is very large and you want to process on the fly
        # Needs adjustment to build the full map first if not streaming
        swe_bm25 = load_dataset(SWE_BENCH_BM25_DATASET)
        # Create a lookup map from instance_id to text content from the BM25 test set
        # Ensure we only use the test split here as well
        text_context_map = {item['instance_id']: item['text'] for item in tqdm(swe_bm25['test'], desc="Building text context map")}
        logging.info(f"Built text context map for {len(text_context_map)} instances.")
    except Exception as e:
        logging.error(f"Failed to load SWE-bench BM25 dataset: {e}")
        exit(1)

    # --- Initialize Models ---
    models_to_test = []

    # Ollama (Llama 3.2)
    try:
        ollama_client = ollama.Client()
        # Basic check to see if Ollama server is reachable
        ollama_client.list()
        logging.info(f"Ollama client initialized. Will use model: {LLAMA_MODEL_NAME}")
        # models_to_test.append({"name": LLAMA_MODEL_NAME, "type": "ollama", "client": ollama_client})
    except Exception as e:
        logging.error(f"Failed to initialize Ollama client or connect to server: {e}")
        logging.warning("Skipping Ollama model.")

    # Gemini
    if not GOOGLE_API_KEY:
        logging.warning("GOOGLE_API_KEY environment variable not set. Skipping Gemini model.")
    else:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            # Make a simple test call (optional, but good for early failure)
            # gemini_model.generate_content("Test", generation_config=genai.types.GenerationConfig(candidate_count=1))
            logging.info(f"Google AI client configured. Will use model: {GEMINI_MODEL_NAME}")
            models_to_test.append({"name": GEMINI_MODEL_NAME, "type": "gemini", "client": gemini_model})
        except Exception as e:
            logging.error(f"Failed to initialize Google AI client/model: {e}")
            logging.warning("Skipping Gemini model.")

    if not models_to_test:
        logging.error("No models available to test. Exiting.")
        exit(1)

    # --- Process Instances ---
    all_predictions = []
    processed_ids = set()
    if OUTPUT_FILE.exists():
        # logging.warning(f"Output file {OUTPUT_FILE} already exists. Appending results.")
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing_preds = [json.loads(line) for line in f]
            processed_ids = {(p['instance_id'], p['model_name_or_path']) for p in existing_preds}
            logging.info(f"Loaded {len(existing_preds)} existing predictions.")
        except Exception as e:
            logging.warning(f"Could not load existing predictions from {OUTPUT_FILE}: {e}")
            processed_ids = set()
    else:
        processed_ids = set()
        pass


    with open(OUTPUT_FILE, 'a') as f_out:
        # Iterate through models first
        for model_info in models_to_test:
            model_name = model_info["name"]
            model_type = model_info["type"]
            client_or_model = model_info["client"]
            logging.info(f"\n--- Processing model: {model_name} ---")

            # Iterate through SWE-bench Lite test instances
            for instance in tqdm(swe_lite_test, desc=f"Generating patches for {model_name}"):
                instance_id = instance['instance_id']


                if processed_ids is not None and (instance_id, model_name) in processed_ids:
                    logging.info(f"Skipping already processed: {instance_id} for {model_name}")
                    continue

                # Get context
                text_context = text_context_map.get(instance_id)
                if not text_context:
                    logging.warning(f"Text context not found for instance_id: {instance_id}. Skipping.")
                    continue

                # Build prompt
                prompt = build_prompt(text_context)

                # Get patch prediction
                model_patch = None
                if model_type == "ollama":
                    model_patch = get_ollama_patch(client_or_model, model_name, prompt)
                elif model_type == "gemini":
                    model_patch = get_gemini_patch(client_or_model, prompt)

                # Handle cases where patch generation failed after retries
                if model_patch is None:
                    logging.error(f"Failed to generate patch for {instance_id} with {model_name} after retries.")
                    model_patch = "" # Store empty patch on total failure

                # Record prediction
                prediction = {
                    "instance_id": instance_id,
                    "model_patch": model_patch,
                    "model_name_or_path": model_name # Use the specific model name
                }
                all_predictions.append(prediction)

                # Write prediction to file immediately (JSON Lines format)
                f_out.write(json.dumps(prediction) + '\n')
                f_out.flush() # Ensure it's written to disk

    logging.info(f"\n--- Benchmark Generation Complete ---")
    logging.info(f"Total predictions generated: {len(all_predictions)}")
    logging.info(f"Results saved to: {OUTPUT_FILE}")