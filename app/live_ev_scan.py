import yfinance as yf
import pandas as pd
import numpy as np
import joblib

# ===============================
# 설정
# ===============================
TICKERS = [
    "SOXL","BULZ","TQQQ","TECL","WEBL","UPRO",
    "WANT","HIBL","FNGU","TNA","RETL","UDOW",
    "NAIL","LABU","PILL","MIDU","CURE","FAS",
    "TPOR","DRN","DUSL","DFEN","UTSL","BNKU","DPST"
]


MODEL_PATH = "app/model.pkl"
SCALER_PATH = "app/scaler.pkl"

FEATURES = [
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio"
]

SUCCESS_RETURN = 0.10
FAIL_MEAN = -0.0179   # 테스트에서 계산한 실패 평균

# ===============================
# 모델 로드
# ===============================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

results = []

for ticker in TICKERS:
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)

        if len(df) < 260:
            continue

        close = df["Close"]

        # ===============================
        # Feature 계산
        # ===============================
        drawdown_252 = close.iloc[-1] / close.rolling(252).max().iloc[-1] - 1
        drawdown_60 = close.iloc[-1] / close.rolling(60).max().iloc[-1] - 1

        atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
        atr_ratio = atr / close.iloc[-1]

        z_score = (
            close.iloc[-1]
            - close.rolling(20).mean().iloc[-1]
        ) / close.rolling(20).std().iloc[-1]

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_hist = (ema12 - ema26).iloc[-1]

        ma20_slope = (
            close.rolling(20).mean().iloc[-1]
            - close.rolling(20).mean().iloc[-5]
        )

        # 시장 기준은 SPY로 가정
        spy = yf.download("SPY", period="2y", interval="1d", progress=False)
        spy_close = spy["Close"]

        market_dd = spy_close.iloc[-1] / spy_close.rolling(252).max().iloc[-1] - 1
        market_atr = (spy["High"] - spy["Low"]).rolling(14).mean().iloc[-1]
        market_atr_ratio = market_atr / spy_close.iloc[-1]

        feature_row = pd.DataFrame([[
            drawdown_252,
            drawdown_60,
            atr_ratio,
            z_score,
            macd_hist,
            ma20_slope,
            market_dd,
            market_atr_ratio
        ]], columns=FEATURES)

        # ===============================
        # 예측
        # ===============================
        X_scaled = scaler.transform(feature_row)
        prob = model.predict_proba(X_scaled)[0, 1]

        ev = prob * SUCCESS_RETURN + (1 - prob) * FAIL_MEAN

        results.append({
            "Ticker": ticker,
            "Prob": round(prob, 4),
            "EV": round(ev, 4)
        })

    except Exception as e:
        print(f"{ticker} error:", e)

# ===============================
# 결과 출력
# ===============================
result_df = pd.DataFrame(results)
result_df = result_df.sort_values("EV", ascending=False)

print("=" * 60)
print("📊 LIVE EV SCAN")
print(result_df)
print("=" * 60)
