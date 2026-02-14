import yfinance as yf
import pandas as pd
from config import *
from universe import *
from indicators import *

rows = []

for ticker in UNIVERSE:

    df = yf.download(ticker, period=f"{LOOKBACK_YEARS}y", progress=False)

    if len(df) < 300:
        continue

    close = df["Close"]
    rsi = compute_rsi(close)
    atr = compute_atr(df)

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()

    for i in range(200, len(df) - DCA_DAYS):

        buy_window = close.iloc[i:i+DCA_DAYS]
        avg_price = buy_window.mean()
        max_price = buy_window.max()

        ret = max_price / avg_price - 1
        success = int(ret >= TARGET)

        rows.append({
            "Date": df.index[i],
            "Ticker": ticker,
            "Close": close.iloc[i],
            "RSI": rsi.iloc[i],
            "Drawdown_60": close.iloc[i] / close.iloc[i-60:i].max() - 1,
            "MA20_above_MA60": int(ma20.iloc[i] > ma60.iloc[i]),
            "MA60_above_MA120": int(ma60.iloc[i] > ma120.iloc[i]),
            "ATR_ratio": atr.iloc[i] / close.iloc[i],
            "Success": success
        })

dataset = pd.DataFrame(rows).dropna()
dataset.to_csv("data/success_dataset.csv", index=False)

print("✅ dataset 생성 완료")
