import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ===============================
# 설정
# ===============================
TICKERS = [
    "SOXL","BULZ","WEBL","LABU","TECL","HIBL","NAIL","DPST",
    "TNA","TQQQ","PILL","DFEN","RETL","TPOR","UTSL",
    "FAS","WANT","DRN","MIDU","DUSL","UDOW","CURE"
]
print("Scaler feature names:")
print(scaler.feature_names_in_)
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

OUTPUT_PATH = "data/backtest_panel.csv"

# ===============================
# 모델 로드
# ===============================
model = joblib.load("app/model.pkl")
scaler = joblib.load("app/scaler.pkl")

# ===============================
# SPY (시장 기준)
# ===============================
print("Downloading SPY...")
spy = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)

if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)

spy = spy.reset_index()

spy["Market_Rolling_Max"] = spy["Close"].rolling(252).max()
spy["Market_Drawdown"] = spy["Close"] / spy["Market_Rolling_Max"] - 1

spy_atr = (spy["High"] - spy["Low"]).rolling(14).mean()
spy["Market_ATR_ratio"] = spy_atr / spy["Close"]

spy["Market_above_MA200"] = (
    spy["Close"] > spy["Close"].rolling(200).mean()
).astype(int)

spy_features = spy[[
    "Date",
    "Market_Drawdown",
    "Market_ATR_ratio",
    "Market_above_MA200"
]]

# ===============================
# 종목 데이터
# ===============================
all_data = []

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

    if df.empty:
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Ticker"] = ticker

    # ===== Feature 계산 =====
    rolling_max_252 = df["Close"].rolling(252).max()
    df["Drawdown_252"] = df["Close"] / rolling_max_252 - 1

    rolling_max_60 = df["Close"].rolling(60).max()
    df["Drawdown_60"] = df["Close"] / rolling_max_60 - 1

    df["Max_Drawdown"] = df["Drawdown_252"]

    mean_20 = df["Close"].rolling(20).mean()
    std_20 = df["Close"].rolling(20).std()
    df["Z_score"] = (df["Close"] - mean_20) / std_20

    df["MA20"] = mean_20
    df["MA20_slope"] = df["MA20"].pct_change(5)

    atr = (df["High"] - df["Low"]).rolling(14).mean()
    df["ATR_ratio"] = atr / df["Close"]

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["MACD_hist"] = macd - signal

    # ===== SPY 병합 =====
    df = df.merge(spy_features, on="Date", how="left")

    # ===== 모델 입력 feature =====
    feature_cols = [
        "Max_Drawdown",
        "Drawdown_60",
        "Drawdown_252",
        "Z_score",
        "ATR_ratio",
        "MACD_hist",
        "MA20_slope",
        "Market_Drawdown",
        "Market_ATR_ratio",
        "Market_above_MA200",
    ]

    df = df.dropna()

    X = df[feature_cols]
    X_scaled = scaler.transform(X)

    # ===== 확률 예측 =====
    df["Pred_Prob"] = model.predict_proba(X_scaled)[:, 1]

    # ===== EV 계산 =====
    df["EV"] = df["Pred_Prob"] * 0.10 + (1 - df["Pred_Prob"]) * (-0.0179)

    all_data.append(df)

# ===============================
# 저장
# ===============================
final_df = pd.concat(all_data).sort_values(["Date", "Ticker"])
final_df.to_csv(OUTPUT_PATH, index=False)

print("✅ backtest_panel.csv 저장 완료 (모델 적용)")
