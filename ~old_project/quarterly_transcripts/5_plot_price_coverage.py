import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Directories for earnings calls and price history
earning_calls_dir = "data/earning_calls"
price_history_dir = "data/price_history"

# Get list of companies from the earnings calls folder (assumed to be the 50ish tickers)
companies = [d for d in os.listdir(earning_calls_dir) if os.path.isdir(os.path.join(earning_calls_dir, d))]

coverage_data = []

# For each company, try to read the corresponding price history file
for ticker in companies:
    file_path = os.path.join(price_history_dir, ticker, f"{ticker}_D_30Y.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, parse_dates=["Date"])
            if not df.empty:
                start_date = df["Date"].min()
                end_date = df["Date"].max()
                coverage_data.append({
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date
                })
            else:
                print(f"No data in price history file for {ticker}.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    else:
        print(f"Price history file not found for {ticker}.")

# Create a DataFrame with the coverage information
coverage_df = pd.DataFrame(coverage_data)

# Sort companies by the start_date for a cleaner display (optional)
coverage_df = coverage_df.sort_values(by="start_date").reset_index(drop=True)

# Plotting a horizontal bar (Gantt chart style) for each company's price history coverage
fig, ax = plt.subplots(figsize=(12, max(6, len(coverage_df)*0.4)))

# For each company, draw a horizontal line spanning from start_date to end_date
for i, row in coverage_df.iterrows():
    ax.hlines(y=i, xmin=row["start_date"], xmax=row["end_date"], color="blue", linewidth=6)
    # Optionally, add markers for the start (green) and end (red)
    ax.plot(row["start_date"], i, "o", color="green")
    ax.plot(row["end_date"], i, "o", color="red")

# Set y-ticks to display company tickers
ax.set_yticks(range(len(coverage_df)))
ax.set_yticklabels(coverage_df["ticker"])

# Format x-axis to display years
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.YearLocator(5))  # tick every 5 years
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.xlabel("Date")
plt.ylabel("Company")
plt.title("30-Year Price History Coverage")
plt.tight_layout()
plt.show()
