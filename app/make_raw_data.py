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
# 🔥 전략 시뮬레이션 (무제한 추가매수 버전)
# ===============================
def simulate_strategy(prices):

    invested = []
    mdd = 0

    # -------------------------
    # 1차 40일 분할매수
    # -------------------------
    success_40 = 0
    exit_day = 39

    for d in range(min(40, len(prices))):
        invested.append(prices[d])
        avg_price = np.mean(invested)
        ret = prices[d] / avg_price - 1
        mdd = min(mdd, ret)

        if ret >= 0.10:
            success_40 = 1
            exit_day = d
            break

    # 🔥 수정: 40일 데이터 없으면 무효 처리
    if len(prices) < 40:
        return None

    # =========================
    # 전략 1 (기존 동일)
    # =========================
    if success_40 == 1:
        strat1_return = 0.10
        strat1_success = 1
        strat1_holding = exit_day + 1
    else:
        avg_40 = np.mean(prices[:40])
        ret_40 = prices[39] / avg_40 - 1
        strat1_return = ret_40
        strat1_success = 0
        strat1_holding = 40

    # =========================
    # 전략 2
    # =========================
    if success_40 == 1:
        strat2_return = 0.10
        strat2_success = 1
        strat2_holding = exit_day + 1

    else:
        avg_40 = np.mean(prices[:40])
        ret_40 = prices[39] / avg_40 - 1

        # 🔵 40일 종료 시 -10% 이내면 그냥 정리
        if ret_40 >= -0.10:
            strat2_return = ret_40
            strat2_success = 0
            strat2_holding = 40

        # 🔴 -10% 초과 손실이면 무제한 추가매수
        else:
            extended_exit = False

            # 🔥 수정: 80일 제한 제거 → 데이터 끝까지
            for d2 in range(40, len(prices)):
                invested.append(prices[d2])
                avg_ext = np.mean(invested)
                ret_ext = prices[d2] / avg_ext - 1
                mdd = min(mdd, ret_ext)

                # 🔥 수정: 평균단가 대비 -10% 회복 시 종료
                if ret_ext >= -0.10:
                    strat2_return = ret_ext
                    strat2_success = 0
                    strat2_holding = d2 + 1
                    extended_exit = True
                    break

            # 🔥 수정: 데이터 끝까지 도달한 경우
            if not extended_exit:
                avg_ext = np.mean(invested)
                final_ret = prices[-1] / avg_ext - 1
                strat2_return = final_ret
                strat2_success = 0
                strat2_holding = len(prices)

    return {
        "Strategy1_Return": strat1_return,
        "Strategy1_Success": strat1_success,
        "Strategy1_Holding": strat1_holding,
        "Strategy2_Return": strat2_return,
        "Strategy2_Success": strat2_success,
        "Strategy2_Holding": strat2_holding,
        "Max_Holding": max(strat1_holding, strat2_holding),
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

    # 🔥 수정: 80일 제한 제거 → 미래 전체 사용
    for i in range(252, len(df) - 1):

        date = df.index[i]
        if date not in market_df.index:
            continue

        prices = close.iloc[i:].values  # 🔥 수정: 끝까지 전달
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

print("✅ raw_data.csv 생성 완료 (전략2 무제한 추가매수 반영)")
