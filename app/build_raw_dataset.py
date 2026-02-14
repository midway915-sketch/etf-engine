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
# 보조지표 함수
# ===============================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

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
# ★ 추가: 전략 시뮬레이션 함수
# ===============================
def simulate_strategy(close, start_idx):

    invested = []
    max_total_days = 200  # 안전 제한
    success_1st = 0
    second_phase = 0
    mdd = 0

    # -------------------------
    # 1차 40일 분할매수
    # -------------------------
    for d in range(40):

        idx = start_idx + d
        if idx >= len(close):
            return None

        invested.append(close.iloc[idx])
        avg_price = np.mean(invested)
        ret = close.iloc[idx] / avg_price - 1
        mdd = min(mdd, ret)

        if ret >= 0.10:
            return {
                "Success_1st": 1,
                "Return_1st": ret,
                "Hold_days_1st": d+1,
                "Second_phase_used": 0,
                "Return_final": ret,
                "Hold_days_final": d+1,
                "Max_hold_days": d+1,
                "Max_drawdown": mdd
            }

    # -------------------------
    # 1차 실패
    # -------------------------
    avg_price = np.mean(invested)
    current_price = close.iloc[start_idx + 39]
    ret_1st = current_price / avg_price - 1
    mdd = min(mdd, ret_1st)

    # -10% 이내면 그냥 종료
    if ret_1st >= -0.10:
        return {
            "Success_1st": 0,
            "Return_1st": ret_1st,
            "Hold_days_1st": 40,
            "Second_phase_used": 0,
            "Return_final": ret_1st,
            "Hold_days_final": 40,
            "Max_hold_days": 40,
            "Max_drawdown": mdd
        }

    # -------------------------
    # 2차 추가매수
    # -------------------------
    second_phase = 1

    for d in range(40, max_total_days):

        idx = start_idx + d
        if idx >= len(close):
            break

        invested.append(close.iloc[idx])
        avg_price = np.mean(invested)
        ret = close.iloc[idx] / avg_price - 1
        mdd = min(mdd, ret)

        if ret >= -0.10:
            return {
                "Success_1st": 0,
                "Return_1st": ret_1st,
                "Hold_days_1st": 40,
                "Second_phase_used": 1,
                "Return_final": ret,
                "Hold_days_final": d+1,
                "Max_hold_days": d+1,
                "Max_drawdown": mdd
            }

    # 강제 종료
    final_idx = min(len(close)-1, start_idx + max_total_days)
    final_price = close.iloc[final_idx]
    avg_price = np.mean(invested)
    final_ret = final_price / avg_price - 1

    return {
        "Success_1st": 0,
        "Return_1st": ret_1st,
        "Hold_days_1st": 40,
        "Second_phase_used": 1,
        "Return_final": final_ret,
        "Hold_days_final": final_idx - start_idx,
        "Max_hold_days": final_idx - start_idx,
        "Max_drawdown": mdd
    }

# ===============================
# Market 데이터
# ===============================
market_df = yf.download(MARKET_TICKER, start=START_DATE, end=END)
market_df.columns = market_df.columns.get_level_values(0)
market_df = market_df.dropna()

market_rsi = rsi(market_df["Close"])
market_drawdown = market_df["Close"] / market_df["Close"].rolling(252).max() - 1
market_atr_ratio = atr(market_df) / market_df["Close"]
market_ma200 = market_df["Close"].rolling(200).mean()

# ===============================
# ETF 데이터 수집
# ===============================
rows = []

for ticker in TICKERS:

    df = yf.download(ticker, start=START_DATE, end=END)
    df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    if len(df) < 350:
        continue

    close = df["Close"]

    # 지표 계산 (기존 동일)
    drawdown_60 = close / close.rolling(60).max() - 1
    drawdown_252 = close / close.rolling(252).max() - 1
    z_score = zscore(close, 60)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_pos = (close - bb_mid) / (2 * bb_std)
    bb_width = (2 * bb_std) / bb_mid

    atr_ratio = atr(df) / close
    realized_vol = close.pct_change().rolling(20).std()

    rsi_val = rsi(close)
    rsi_slope = rsi_val.diff(5)

    roc5 = close.pct_change(5)
    roc10 = close.pct_change(10)
    roc20 = close.pct_change(20)

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_hist = ema12 - ema26

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()

    ma20_gap = (close - ma20) / ma20
    ma60_gap = (close - ma60) / ma60
    ma120_gap = (close - ma120) / ma120
    ma20_slope = ma20.diff(5)

    volume_ratio = df["Volume"] / df["Volume"].rolling(20).mean()
    obv = (np.sign(close.diff()) * df["Volume"]).fillna(0).cumsum()
    obv_change = obv.pct_change(5)

    for i in range(252, len(df) - 200):  # ★ 수정: 2차 대비 200일 확보

        date = df.index[i]
        if date not in market_df.index:
            continue

        m_idx = market_df.index.get_loc(date)

        # ★ 추가: 전략 시뮬레이션 실행
        sim = simulate_strategy(close, i)
        if sim is None:
            continue

        rows.append({
            "Ticker": ticker,

            # ★ 추가: 전략 결과
            "Success_1st": sim["Success_1st"],
            "Return_1st": sim["Return_1st"],
            "Hold_days_1st": sim["Hold_days_1st"],
            "Second_phase_used": sim["Second_phase_used"],
            "Return_final": sim["Return_final"],
            "Hold_days_final": sim["Hold_days_final"],
            "Max_hold_days": sim["Max_hold_days"],
            "Max_drawdown": sim["Max_drawdown"],

            # 기존 지표
            "Drawdown_60": drawdown_60.iloc[i],
            "Drawdown_252": drawdown_252.iloc[i],
            "Z_score": z_score.iloc[i],
            "BB_position": bb_pos.iloc[i],
            "ATR_ratio": atr_ratio.iloc[i],
            "BB_width": bb_width.iloc[i],
            "Realized_vol": realized_vol.iloc[i],
            "RSI": rsi_val.iloc[i],
            "RSI_slope": rsi_slope.iloc[i],
            "ROC_5": roc5.iloc[i],
            "ROC_10": roc10.iloc[i],
            "ROC_20": roc20.iloc[i],
            "MACD_hist": macd_hist.iloc[i],
            "MA20_gap": ma20_gap.iloc[i],
            "MA60_gap": ma60_gap.iloc[i],
            "MA120_gap": ma120_gap.iloc[i],
            "MA20_slope": ma20_slope.iloc[i],
            "Volume_ratio": volume_ratio.iloc[i],
            "OBV_change": obv_change.iloc[i],
            "Market_RSI": market_rsi.iloc[m_idx],
            "Market_Drawdown": market_drawdown.iloc[m_idx],
            "Market_ATR_ratio": market_atr_ratio.iloc[m_idx],
            "Market_above_MA200": int(
                market_df["Close"].iloc[m_idx] > market_ma200.iloc[m_idx]
            ),
        })

raw_df = pd.DataFrame(rows)
raw_df = raw_df.dropna()
raw_df.to_csv("data/raw_data.csv", index=False)

print("✅ 1차 + 2차 전략 포함 raw_data.csv 생성 완료")
