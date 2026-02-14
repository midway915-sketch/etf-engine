import pandas as pd
import numpy as np
from config import *

df = pd.read_csv("data/scored_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

def run_backtest(threshold, mode="A"):

    capital = INITIAL_CAPITAL
    trades = 0
    wins = 0
    holding_days = []

    i = 0

    while i < len(df) - DCA_DAYS:

        row = df.iloc[i]

        if row["Probability"] >= threshold:

            trades += 1
            total_invest = 0
            total_shares = 0
            start_index = i

            # -----------------------
            # 1차 40일 분할매수
            # -----------------------
            sold = False

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
                    holding_days.append(j+1)
                    i += j+1
                    sold = True
                    break

            if sold:
                continue

            # -----------------------
            # 40일 종료 시점
            # -----------------------

            final_price = df.iloc[i + DCA_DAYS - 1]["Close"]
            avg_price = total_invest / total_shares
            ret = final_price / avg_price - 1

            # ===== 전략 A =====
            if mode == "A":

                capital *= (1 + ret)
                holding_days.append(DCA_DAYS)
                i += DCA_DAYS
                continue

            # ===== 전략 B =====
            if mode == "B":

                # -10% 이하 손실이면 정리
                if ret >= STOP_LOSS:
                    capital *= (1 + ret)
                    holding_days.append(DCA_DAYS)
                    i += DCA_DAYS
                    continue

                # -10% 초과 손실 → 무한 추가매수
                j = DCA_DAYS

                while i + j < len(df):

                    price = df.iloc[i+j]["Close"]
                    invest = capital / DCA_DAYS

                    total_invest += invest
                    total_shares += invest / price

                    avg_price = total_invest / total_shares
                    ret = price / avg_price - 1

                    if ret >= STOP_LOSS:
                        capital *= (1 + STOP_LOSS)
                        holding_days.append(j+1)
                        break

                    j += 1

                i += j+1
                continue

        else:
            i += 1

    return {
        "Threshold": threshold,
        "Mode": mode,
        "Trades": trades,
        "WinRate": round(wins/trades,4) if trades else 0,
        "FinalCapital": int(capital),
        "TotalMultiple": round(capital/INITIAL_CAPITAL,2),
        "AvgHoldingDays": round(np.mean(holding_days),2) if holding_days else 0,
        "MaxHoldingDays": max(holding_days) if holding_days else 0
    }

results = []

for th in THRESHOLDS:
    results.append(run_backtest(th, "A"))
    results.append(run_backtest(th, "B"))

result_df = pd.DataFrame(results)
print(result_df)

result_df.to_csv("data/backtest_results.csv", index=False)



for th in THRESHOLDS:
    print(run_backtest(th, "A"))
    print(run_backtest(th, "B"))
    print("-"*60)
