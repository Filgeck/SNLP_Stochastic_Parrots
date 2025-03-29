import eikon as ek
import pandas as pd
import os
from datetime import datetime, timedelta

# Set your Eikon API key
ek.set_app_key(os.environ.get("EIKON_API_KEY"))

# Load S&P500 constituents from the CSV generated earlier
constituents_file = 'data/snp500_constituents.csv'
df_constituents = pd.read_csv(constituents_file)

# Define the date range for the last 30 years
end_date = datetime.now()
start_date = end_date - timedelta(days=30*365)  # Approximate 30 years; adjust if necessary
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Loop over each company
for index, row in df_constituents.iterrows():
    ticker = row["Ticker Symbol"].strip()  # Clean up any extra whitespace
    print(f"Downloading daily price data for {ticker} from {start_date_str} to {end_date_str}...")
    
    try:
        # Download daily price data using Eikon's get_timeseries function.
        # The 'interval' parameter is set to 'daily' for daily data.
        price_data = ek.get_timeseries(row["Instrument"].strip(), start_date=start_date_str, end_date=end_date_str, interval='daily')
        
        # Create a folder for this ticker under data/price_history
        ticker_folder = os.path.join("data/price_history", ticker)
        os.makedirs(ticker_folder, exist_ok=True)
        
        # Save the price data to a CSV file named {ticker}_D_30Y.csv
        file_path = os.path.join(ticker_folder, f"{ticker}_D_30Y.csv")
        price_data.to_csv(file_path)
        
        print(f"Saved price data for {ticker} to {file_path}")
    
    except Exception as e:
        print(f"Error downloading price data for {ticker}: {e}")
