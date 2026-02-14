import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ===============================
# 설정
# ===============================
TICKERS = [
    "SOXL","BULZ","WEBL","LABU","TECL","HIBL","NAIL","DPST",
    "TNA","TQQQ","PILL","DFEN","RETL","TPOR","UTSL",
    "FAS","WANT","DRN","MIDU","DUSL","UDOW","CURE","TSLL"
]

START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

OUTPUT_PATH = "data/backtest_panel.csv"

# ===============================
# 데이터 다운로드
# ===============================
all_data = []

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

    if df.empty:
        continue

    df = df.reset_index()
    df["Ticker"] = ticker

    # ---------------------------
    # Feature 계산 (간단 버전)
    # ---------------------------
    df["Return"] = df["Close"].pct_change()

    # Drawdown
    rolling_max = df["Close"].rolling(252).max()
    df["Drawdown_252"] = df["Close"] / rolling_max - 1

    rolling_max_60 = df["Close"].rolling(60).max()
    df["Drawdown_60"] = df["Close"] / rolling_max_60 - 1

    df["Max_Drawdown"] = df["Drawdown_252"]

    # Z-score
    mean_20 = df["Close"].rolling(20).mean()
    std_20 = df["Close"].rolling(20).std()
    df["Z_score"] = (df["Close"] - mean_20) / std_20

    # MA slope
    df["MA20"] = mean_20
    df["MA20_slope"] = df["MA20"].pct_change(5)

    # ATR ratio
    high_low = df["High"] - df["Low"]
    atr = high_low.rolling(14).mean()
    df["ATR_ratio"] = atr / df["Close"]

    # MACD histogram
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["MACD_hist"] = macd - signal

    # Market proxy (SPY)
    # 간단히 자기 자신 기준으로 대체 (원하면 SPY 따로 붙일 수 있음)
    df["Market_Drawdown"] = df["Drawdown_252"]
    df["Market_ATR_ratio"] = df["ATR_ratio"]
    df["Market_above_MA200"] = (df["Close"] > df["Close"].rolling(200).mean()).astype(int)

    df = df.dropna()

    all_data.append(df)

# ===============================
# 통합 저장
# ===============================
final_df = pd.concat(all_data).sort_values(["Date", "Ticker"])
final_df.to_csv(OUTPUT_PATH, index=False)

print("✅ backtest_panel.csv 저장 완료")
