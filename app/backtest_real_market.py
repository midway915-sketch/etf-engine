import pandas as pd
import numpy as np
from config import *

df = pd.read_csv("data/scored_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

def run_backtest(threshold):

    capital = INITIAL_CAPITAL
    position = False
    trades = 0
    wins = 0
    holding_days = []

    i = 0

    while i < len(df) - DCA_DAYS:

        if (not position) and df.iloc[i]["Probability"] >= threshold:

            position = True
            trades += 1
            start = i

            total_invest = 0
            total_shares = 0

            for j in range(DCA_DAYS):

                price = df.iloc[i+j]["Close"]
                invest = capital / DCA_DAYS
                total_invest += invest
                total_shares += invest / price

                avg_price = total_invest / total_shares
                ret = price / avg_price - 1

                if ret >= TARGET:
                    capital *= (1 + TARGET)
                    wins += 1
                    break

            else:
                final_price = df.iloc[i+DCA_DAYS-1]["Close"]
                ret = final_price / avg_price - 1
                capital *= (1 + ret)

            holding_days.append(j+1)
            position = False
            i += DCA_DAYS

        else:
            i += 1

    return {
        "Threshold": threshold,
        "Trades": trades,
        "WinRate": round(wins/trades,4) if trades else 0,
        "FinalCapital": int(capital),
        "TotalMultiple": round(capital/INITIAL_CAPITAL,2),
        "AvgHoldingDays": round(np.mean(holding_days),2) if holding_days else 0,
        "MaxHoldingDays": max(holding_days) if holding_days else 0
    }

for th in THRESHOLDS:
    print(run_backtest(th))
