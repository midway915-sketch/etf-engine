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
# 🔥 전략 시뮬레이션 (용어 변경 반영)
# ===============================
def simulate_strategy(prices):

    invested = []
    mdd = 0

    # -------------------------
    # 1차 40일 분할매수
    # -------------------------
    success = 0   # 🔥 수정: Success 단일 정의
    exit_day = 39

    for d in range(min(40, len(prices))):
        invested.append(prices[d])
        avg_price = np.mean(invested)
        ret = prices[d] / avg_price - 1
        mdd = min(mdd, ret)

        if ret >= 0.10:
            success = 1   # 🔥 수정
            exit_day = d
            break

    if len(prices) < 40:
        return None

    # =========================
    # 🔥 수정: 실패1 (40일차 정리)
    # =========================
    avg_40 = np.mean(prices[:40])
    ret_40 = prices[39] / avg_40 - 1
    return_fail1 = ret_40   # 🔥 수정: Return_Fail1

    # =========================
    # 🔥 수정: 실패2 (확장형)
    # =========================
    if success == 1:
        return_fail2 = 0.10  # 성공 시 동일하게 10%
        holding = exit_day + 1
    else:
        if ret_40 >= -0.10:
            return_fail2 = ret_40
            holding = 40
        else:
            extended_exit = False

            for d2 in range(40, len(prices)):
                invested.append(prices[d2])
                avg_ext = np.mean(invested)
                ret_ext = prices[d2] / avg_ext - 1
                mdd = min(mdd, ret_ext)

                if ret_ext >= -0.10:
                    return_fail2 = ret_ext
                    holding = d2 + 1
                    extended_exit = True
                    break

            if not extended_exit:
                avg_ext = np.mean(invested)
                return_fail2 = prices[-1] / avg_ext - 1
                holding = len(prices)

    return {
        "Success": success,                 # 🔥 수정
        "Return_Fail1": return_fail1,       # 🔥 수정
        "Return_Fail2": return_fail2,       # 🔥 수정
        "Holding_Period": holding,
        "Max_Drawdown": mdd
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
# 데이터 생성
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

    for i in range(252, len(df) - 1):

        date = df.index[i]
        if date not in market_df.index:
            continue

        prices = close.iloc[i:].values
        sim = simulate_strategy(prices)

        if sim is None:
            continue

        m_idx = market_df.index.get_loc(date)

        rows.append({
            "Date": date,
            "Ticker": ticker,
            **sim,

            # 학습용 피처
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

print("✅ raw_data.csv 생성 완료 (Success / Fail1 / Fail2 구조 적용)")
