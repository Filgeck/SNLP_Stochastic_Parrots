import os
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# Helper functions
# ---------------------------

def generate_quarter_list():
    """
    Generate a list of (year, quarter) tuples from Q3 2013 to Q3 2024 inclusive.
    Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec.
    """
    quarters = []
    start_year, start_q = 2013, 3
    end_year, end_q = 2024, 3
    year = start_year
    quarter = start_q
    while (year < end_year) or (year == end_year and quarter <= end_q):
        quarters.append((year, quarter))
        if quarter == 4:
            quarter = 1
            year += 1
        else:
            quarter += 1
    return quarters

def quarter_end_date(year, quarter):
    """
    Return a pd.Timestamp corresponding to the quarter end date.
    """
    if quarter == 1:
        return pd.Timestamp(f"{year}-03-31")
    elif quarter == 2:
        return pd.Timestamp(f"{year}-06-30")
    elif quarter == 3:
        return pd.Timestamp(f"{year}-09-30")
    elif quarter == 4:
        return pd.Timestamp(f"{year}-12-31")

def has_all_transcripts(ticker, quarters):
    """
    Check if a given ticker has transcript files for every quarter in the provided list.
    """
    transcript_folder = os.path.join("data", "earning_calls", ticker)
    if not os.path.exists(transcript_folder):
        return False
    for year, q in quarters:
        transcript_file = os.path.join(transcript_folder, f"{ticker}_{year}_Q{q}.txt")
        if not os.path.exists(transcript_file):
            return False
    return True

def has_full_price_history(ticker, required_start_date, required_end_date):
    """
    Check if a given ticker's price history CSV exists and covers the date range.
    """
    price_folder = os.path.join("data", "price_history", ticker)
    csv_file = os.path.join(price_folder, f"{ticker}_D_30Y.csv")
    if not os.path.exists(csv_file):
        return False
    try:
        df = pd.read_csv(csv_file, parse_dates=["Date"])
    except Exception as e:
        print(f"Error reading price history for {ticker}: {e}")
        return False
    if df.empty:
        return False
    if df["Date"].min() > required_start_date or df["Date"].max() < required_end_date:
        return False
    return True

def get_closing_price(df, target_date):
    """
    Given a sorted DataFrame of price history, find the closing price on the target_date.
    If no exact match exists, return the first available price after the target_date.
    Returns a tuple of (closing price, actual date used) or (None, None) if not found.
    """
    # Exact match:
    exact = df[df["Date"] == target_date]
    if not exact.empty:
        return exact.iloc[0]["CLOSE"], target_date
    # Get the first date after target_date:
    future = df[df["Date"] > target_date]
    if not future.empty:
        next_date = future.iloc[0]["Date"]
        return future.iloc[0]["CLOSE"], next_date
    return None, None

def build_dataset(valid_tickers, quarters):
    """
    For each valid ticker and quarter, load the transcript text and compute the target return.
    The target return is defined as the percentage change from the closing price on (or immediately after)
    the quarter end date to the closing price 7 days later.
    """
    data_rows = []
    for ticker in valid_tickers:
        price_file = os.path.join("data", "price_history", ticker, f"{ticker}_D_30Y.csv")
        try:
            price_df = pd.read_csv(price_file, parse_dates=["Date"])
        except Exception as e:
            print(f"Error loading price history for {ticker}: {e}")
            continue
        price_df = price_df.sort_values("Date")
        
        for year, q in quarters:
            transcript_file = os.path.join("data", "earning_calls", ticker, f"{ticker}_{year}_Q{q}.txt")
            if not os.path.exists(transcript_file):
                continue
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            # Assume the transcript is released at quarter end.
            release_date = quarter_end_date(year, q)
            price_release, actual_release_date = get_closing_price(price_df, release_date)
            if price_release is None:
                continue
            # Find the price 7 days later.
            future_date = actual_release_date + pd.Timedelta(days=7)
            price_after, actual_future_date = get_closing_price(price_df, future_date)
            if price_after is None:
                continue
            target_return = (price_after / price_release) - 1
            data_rows.append({
                "ticker": ticker,
                "year": year,
                "quarter": q,
                "transcript": transcript_text,
                "target_return": target_return,
                "release_date": actual_release_date
            })
    return pd.DataFrame(data_rows)

# ---------------------------
# Main script
# ---------------------------

def main():
    # Define the quarter range and required price history dates.
    quarters = generate_quarter_list()
    required_start_date = pd.Timestamp("2013-07-01")
    required_end_date = pd.Timestamp("2024-09-30")
    
    # Load S&P500 constituents.
    constituents_file = os.path.join("data", "snp500_constituents.csv")
    df_const = pd.read_csv(constituents_file)
    tickers = df_const["Ticker Symbol"].unique()
    
    # Filter tickers to only those with complete transcript and price history data.
    valid_tickers = []
    for ticker in tickers:
        if has_all_transcripts(ticker, quarters) and has_full_price_history(ticker, required_start_date, required_end_date):
            valid_tickers.append(ticker)
    print(f"Found {len(valid_tickers)} valid tickers out of {len(tickers)}")
    
    # Build the modeling dataset.
    dataset = build_dataset(valid_tickers, quarters)
    print(f"Built dataset with {len(dataset)} transcript entries")
    
    # Ensure release_date is in datetime format and sort the dataset.
    dataset["release_date"] = pd.to_datetime(dataset["release_date"])
    dataset.sort_values("release_date", inplace=True)
    
    # Split dataset into training and testing based on time (e.g., before 2021 for training).
    split_date = pd.Timestamp("2021-01-01")
    train_df = dataset[dataset["release_date"] < split_date]
    test_df = dataset[dataset["release_date"] >= split_date]
    print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")
    
    # Use TF-IDF to vectorize transcript texts.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(train_df["transcript"])
    X_test = vectorizer.transform(test_df["transcript"])
    
    y_train = train_df["target_return"].values
    y_test = test_df["target_return"].values
    
    # Train a regression model (Ridge regression).
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate the model.
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print("Training MSE:", mse_train)
    print("Testing MSE:", mse_test)
    print("Training R2:", r2_train)
    print("Testing R2:", r2_test)
    
    # ---------------------------
    # Plotting Results
    # ---------------------------
    
    # Scatter plot: Predicted vs Actual returns on test set.
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.xlabel("Actual Return")
    plt.ylabel("Predicted Return")
    plt.title("Predicted vs. Actual Returns (Test Set)")
    # Plot a diagonal line for reference.
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png")
    plt.show()
    
    # Histogram of residuals.
    residuals = y_test - y_pred_test
    plt.figure(figsize=(8,6))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals (Test Set)")
    plt.tight_layout()
    plt.savefig("residuals_histogram.png")
    plt.show()

if __name__ == "__main__":
    main()
