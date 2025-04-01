import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Directories
earning_calls_dir = "data/earning_calls"
price_history_dir = "data/price_history"

# Get the list of companies based on the earnings call directories
companies = [d for d in os.listdir(earning_calls_dir) if os.path.isdir(os.path.join(earning_calls_dir, d))]
companies.sort()  # sort alphabetically

# Helper function: Convert (year, quarter) to a representative date
def quarter_to_date(year, quarter):
    month = {1: 1, 2: 4, 3: 7, 4: 10}.get(quarter, 1)
    return datetime(year, month, 1)

# Build a dictionary with transcript event dates per company
transcript_dates = {}
for ticker in companies:
    ticker_dir = os.path.join(earning_calls_dir, ticker)
    transcript_dates[ticker] = []
    for filename in os.listdir(ticker_dir):
        if filename.endswith(".txt"):
            # Expected format: TICKER_YEAR_Q{quarter}.txt
            parts = filename.split('_')
            if len(parts) >= 3:
                try:
                    # e.g., for "AAPL_2020_Q3.txt":
                    # parts[0] = "AAPL", parts[1] = "2020", parts[2] = "Q3.txt"
                    year = int(parts[1])
                    quarter_str = parts[2]
                    quarter = int(quarter_str.replace("Q", "").replace(".txt", ""))
                    dt = quarter_to_date(year, quarter)
                    transcript_dates[ticker].append(dt)
                except Exception as e:
                    print(f"Error processing file {filename} for {ticker}: {e}")

# Read price history data and store start and end dates for each company
price_coverage = {}  # mapping: ticker -> (start_date, end_date)
for ticker in companies:
    file_path = os.path.join(price_history_dir, ticker, f"{ticker}_D_30Y.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, parse_dates=["Date"])
            if not df.empty:
                start_date = df["Date"].min()
                end_date = df["Date"].max()
                price_coverage[ticker] = (start_date, end_date)
        except Exception as e:
            print(f"Error reading price history for {ticker}: {e}")
    else:
        print(f"Price history file not found for {ticker}")

# Create a mapping from ticker to a y-axis index for plotting
ticker_to_y = {ticker: i for i, ticker in enumerate(companies)}

# Create the plot
fig, ax = plt.subplots(figsize=(15, max(6, len(companies) * 0.5)))

# Plot price history as a horizontal line per company (blue bar with start/end markers)
for ticker, y in ticker_to_y.items():
    if ticker in price_coverage:
        start_date, end_date = price_coverage[ticker]
        ax.hlines(y=y, xmin=start_date, xmax=end_date, color='blue', linewidth=4, 
                  label='Price Coverage' if y == 0 else "")
        ax.plot(start_date, y, 'o', color='green')  # start marker
        ax.plot(end_date, y, 'o', color='red')      # end marker

# Plot transcript events as scatter points (orange markers)
for ticker, y in ticker_to_y.items():
    dates = transcript_dates.get(ticker, [])
    if dates:
        ax.scatter(dates, [y]*len(dates), color='orange', s=50, zorder=5,
                   label='Transcript Coverage' if y == 0 else "")

# Format the x-axis with date ticks
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Label y-axis with tickers
ax.set_yticks(list(ticker_to_y.values()))
ax.set_yticklabels(list(ticker_to_y.keys()))

plt.xlabel("Date")
plt.ylabel("Company")
plt.title("Price History and Earnings Transcript Coverage")
plt.legend()
plt.tight_layout()
plt.show()
