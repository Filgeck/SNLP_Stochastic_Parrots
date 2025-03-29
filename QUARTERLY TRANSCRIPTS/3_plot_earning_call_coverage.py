import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directory where the earnings call transcripts are saved
data_dir = "data/earning_calls"

# Get list of companies (folder names) within the data directory
companies = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Generate a list of all quarters from 1995 Q1 to 2024 Q4
quarters = []
for year in range(1995, 2025):  # 2025 is not included, so last year is 2024
    for q in range(1, 5):
        quarters.append(f"{year}_Q{q}")

# Create a DataFrame to hold coverage information.
# Rows: companies, Columns: quarters. Initialize all cells to 0 (no transcript)
coverage_df = pd.DataFrame(0, index=companies, columns=quarters)

# Loop through each company's folder and mark available quarters
for company in companies:
    company_folder = os.path.join(data_dir, company)
    for filename in os.listdir(company_folder):
        if filename.endswith(".txt"):
            # Expected filename format: {ticker}_{year}_Q{quarter}.txt (e.g., AAPL_2020_Q3.txt)
            parts = filename.split('_')
            if len(parts) >= 3:
                # Extract year and quarter (remove file extension from quarter part)
                year = parts[1]
                quarter = parts[2].split('.')[0]  # e.g., "Q3"
                col = f"{year}_{quarter}"
                if col in coverage_df.columns:
                    coverage_df.loc[company, col] = 1

# Plot a heatmap of transcript coverage
plt.figure(figsize=(20, max(6, len(companies) * 0.3)))  # Adjust figure height as needed
sns.heatmap(coverage_df, cmap="Greens", cbar=False, linewidths=0.5, linecolor='gray')
plt.title("Earnings Call Transcript Coverage (1995 Q1 - 2024 Q4)")
plt.xlabel("Quarter")
plt.ylabel("Company")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
