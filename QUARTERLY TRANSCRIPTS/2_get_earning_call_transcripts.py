import requests
import pandas as pd
import os
from datetime import datetime
import time

# Set your API key (you can also store this in your environment variables)
FMP_API_KEY = os.environ.get("FMP_API_KEY")  # Ensure you've set your API key in the environment
BASE_URL = "https://financialmodelingprep.com/api/v3/earning_call_transcript"

# Load S&P500 constituents from the CSV generated earlier
constituents_file = 'data/snp500_constituents.csv'
df_constituents = pd.read_csv(constituents_file)

# Create output folder if it doesn't exist
output_folder = "data/earning_calls"
os.makedirs(output_folder, exist_ok=True)

# Define the range for the last 20 years.
current_year = datetime.now().year - 1
start_year = current_year - 18  # adjust as needed if some companies have shorter histories

# Loop over each company
for index, row in df_constituents.iterrows():
    ticker = row["Ticker Symbol"].strip()  # remove any extra whitespace
    print(f"Processing {ticker}...")
    
    # For each year and quarter, attempt to fetch the transcript
    for year in range(start_year, current_year + 1):
        for quarter in range(1, 5):

            # Check if the file already exists
            filename = f"{ticker}_{year}_Q{quarter}.txt"
            filepath = os.path.join(output_folder, f"{ticker}/" + filename)
            if os.path.exists(filepath):
                print(f"Already done for {ticker} {year} Q{quarter}")
                continue

            time.sleep(0.2)

            # Build URL; note that since we already have a query parameter for year and quarter,
            # we append the API key with &apikey=
            url = f"{BASE_URL}/{ticker}?year={year}&quarter={quarter}&apikey={FMP_API_KEY}"
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"Failed for {ticker} {year} Q{quarter}: HTTP {response.status_code}")
                    continue

                data = response.json()
                # The API returns a list; if it contains data, we use the first (or only) element
                if data and isinstance(data, list) and len(data) > 0:
                    transcript_entry = data[0]
                    content = transcript_entry.get("content")
                    if content:
                        # Create a file name using the ticker, year, and quarter
                        filename = f"{ticker}_{year}_Q{quarter}.txt"
                        os.makedirs(output_folder+f"/{ticker}", exist_ok=True)
                        filepath = os.path.join(output_folder, f"{ticker}/"+filename)
                        with open(filepath, "w", encoding="utf-8") as file:
                            file.write(content)
                        print(f"Saved transcript for {ticker} {year} Q{quarter}")
                    else:
                        print(f"No content for {ticker} {year} Q{quarter}")
                else:
                    print(f"No transcript available for {ticker} {year} Q{quarter}")
            except Exception as e:
                print(f"Error for {ticker} {year} Q{quarter}: {e}")
