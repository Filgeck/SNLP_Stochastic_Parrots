import eikon as ek
import pandas as pd
import os

ek.set_app_key(os.environ.get("EIKON_API_KEY"))

# Get the constituents of the S&P 500
snp500, err = ek.get_data('0#.SPX', ['TR.CommonName', 'TR.TickerSymbol'])

if err:
    print(f"Error: {err}")
else:
    df = pd.DataFrame(snp500)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/snp500_constituents.csv', index=False)