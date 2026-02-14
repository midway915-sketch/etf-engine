import pandas as pd
import numpy as np
import yfinance as yf
import os   # 👈 이 줄 추가


df = pd.read_csv("data/raw_data.csv")
df = df.sort_values(["Ticker", "Date"])
df["Date"] = pd.to_datetime(df["Date"])

def compute_success(group):
    group = group.copy()
    group["Future_Max"] = group["Close"].shift(-1).rolling(40).max()
    group["Target"] = (group["Future_Max"] >= group["Close"] * 1.10).astype(int)
    return group

df = df.groupby("Ticker", group_keys=False).apply(compute_success)

# 40일 뒤 데이터 없는 부분 제거
df = df.dropna(subset=["Future_Max"])

# 필요한 컬럼만 유지
df = df[["Date","Ticker","Open","High","Low","Close","Volume","Target"]]

df.to_csv("data/success_dataset.csv", index=False)

print("✅ success_dataset.csv 생성 완료")
