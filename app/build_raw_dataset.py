import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime

# ===============================
# 기본 설정
# ===============================
TICKERS = [
    "SOXL","BULZ","TQQQ","TECL","WEBL","UPRO",
    "WANT","HIBL","FNGU","TNA","RETL","UDOW",
    "NAIL","LABU","PILL","MIDU","CURE","FAS",
    "TPOR","DRN","DUSL","DFEN","UTSL","BNKU","DPST"
]

MARKET_TICKER = "SPY"
START_DATE = "2015-01-01"
END = datetime.today().strftime("%Y-%m-%d")

os.makedirs("data", exist_ok=True)

# ===============================
# 보조지표
# ===============================
def zscore(series, window=60):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ===============================
# 전략 시뮬레이션
# ===============================
def simulate(close, start_idx):

    invested = []
    mdd = 0
    max_total_days = 200

    # -------------------------
    # 1차 40일 분할매수
    # -------------------------
    for d in range(40):
        idx = start_idx + d
        if idx >= len(close):
            return None
        invested.append(close.iloc[idx])
        avg = np.mean(invested)
        ret = close.iloc[idx] / avg - 1
        mdd = min(mdd, ret)
        if ret >= 0.10:
            return {
                "Return_S1": ret,
                "Hold_S1": d+1,
                "Success_S1": 1,
                "Return_S2": ret,
                "Hold_S2": d+1,
                "Success_S2": 1,
                "Max_hold": d+1,
                "MDD": mdd
            }

    # 40일 종료 시점
    avg = np.mean(invested)
    price_40 = close.iloc[start_idx+39]
    ret_40 = price_40 / avg - 1
    mdd = min(mdd, ret_40)

    # -------------------------
    # 전략1: 40일 종료
    # -------------------------
    result_S1 = {
        "Return_S1": ret_40,
        "Hold_S1": 40,
        "Success_S1": int(ret_40 >= 0.10)
    }

    # -------------------------
    # 전략2
    # -------------------------
    if ret_40 >= -0.10:
        result_S2 = {
            "Return_S2": ret_40,
            "Hold_S2": 40,
            "Success_S2": int(ret_40 >= 0.10)
        }
        max_hold = 40
    else:
        # 추가 매수
        for d in range(40, max_total_days):
            idx = start_idx + d
            if idx >= len(close):
                break
            invested.append(close.iloc[idx])
            avg = np.mean(invested)
            ret = close.iloc[idx] / avg - 1
            mdd = min(mdd, ret)
            if ret >= -0.10:
                result_S2 = {
                    "Return_S2": ret,
                    "Hold_S2": d+1,
                    "Success_S2": int(ret >= 0.10)
                }
                max_hold = d+1
                break
        else:
            result_S2 = {
                "Return_S2": ret,
                "Hold_S2": d+1,
                "Success_S2": 0
            }
            max_hold = d+1

    return {
        **result_S1,
        **result_S2,
        "Max_hold": max_hold,
        "MDD": mdd
    }

# ===============================
# Market 데이터
# ===============================
market_df = yf.download(MARKET_TICKER, start=START_DATE, end=END)
market_df.columns = market_df.columns.get_level_values(0)
market_df = market_df.dropna()

market_drawdown = market_df["Close"] / market_df["Close"].rolling(252).max() - 1
market_atr_ratio = atr(market_df) / market_df["Close"]
market_ma200 = market_df["Close"].rolling(200).mean()

# ===============================
# 데이터 수집
# ===============================
rows = []

for ticker in TICKERS:

    df = yf.download(ticker, start=START_DATE, end=END)
    df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    if len(df) < 400:
        continue

    close = df["Close"]

    drawdown_60 = close / close.rolling(60).max() - 1
    drawdown_252 = close / close.rolling(252).max() - 1
    z_score = zscore(close, 60)
    atr_ratio = atr(df) / close

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_hist = ema12 - ema26

    ma20 = close.rolling(20).mean()
    ma20_slope = ma20.diff(5)

    for i in range(252, len(df)-200):

        date = df.index[i]
        if date not in market_df.index:
            continue

        sim = simulate(close, i)
        if sim is None:
            continue

        m_idx = market_df.index.get_loc(date)

        rows.append({
            "Date": date,
            "Ticker": ticker,

            **sim,

            "Drawdown_60": drawdown_60.iloc[i],
            "Drawdown_252": drawdown_252.iloc[i],
            "Z_score": z_score.iloc[i],
            "ATR_ratio": atr_ratio.iloc[i],
            "MACD_hist": macd_hist.iloc[i],
            "MA20_slope": ma20_slope.iloc[i],
            "Market_Drawdown": market_drawdown.iloc[m_idx],
            "Market_ATR_ratio": market_atr_ratio.iloc[m_idx],
            "Market_above_MA200": int(
                market_df["Close"].iloc[m_idx] > market_ma200.iloc[m_idx]
            ),
        })

raw_df = pd.DataFrame(rows)
raw_df = raw_df.dropna()
raw_df = raw_df.sort_values("Date")

raw_df.to_csv("data/raw_data.csv", index=False)

print("✅ 전략1 / 전략2 분리 + Date 포함 raw_data 생성 완료")
