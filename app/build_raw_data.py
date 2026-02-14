import pandas as pd
import yfinance as yf
from datetime import datetime

TICKERS = [
    "SOXL","BULZ","TQQQ","TECL","WEBL","UPRO",
    "WANT","HIBL","FNGU","TNA","RETL","UDOW",
    "NAIL","LABU","PILL","MIDU","CURE","FAS",
    "TPOR","DRN","DUSL","DFEN","UTSL","BNKU","DPST"
]

START = "2020-01-01"
END = datetime.today().strftime("%Y-%m-%d")

all_data = []

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START, end=END, interval="1d", progress=False)

    if df.empty:
        continue

    df = df.reset_index()
    df["Ticker"] = ticker
    all_data.append(df)

raw = pd.concat(all_data)
raw = raw[["Date","Ticker","Open","High","Low","Close","Adj Close","Volume"]]
raw = raw.sort_values(["Ticker","Date"])

raw.to_csv("data/raw_data.csv", index=False)

print("✅ raw_data.csv 생성 완료")
