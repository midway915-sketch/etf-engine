import pandas as pd
import numpy as np

INPUT_PATH = "data/backtest_panel.csv"
OUTPUT_PATH = "data/parametric_results.csv"

INITIAL_SEED = 40_000_000

df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values(["Date", "Ticker"])

ev_quantiles = [0.70, 0.75, 0.80, 0.85, 0.90]
holding_days_list = [30, 40, 50]
stop_levels = [-0.05, -0.10, -0.15]

results = []

def run_backtest(ev_cut, max_days, stop_level):

    seed = INITIAL_SEED
    in_position = False
    idle_days = 0

    entry_price = 0
    total_shares = 0
    total_invested = 0
    holding_day = 0

    equity_curve = []
    max_equity = seed
    max_dd = 0

    daily_amount = INITIAL_SEED / max_days

    grouped = df.groupby("Date")

    for date, day_data in grouped:

        if not in_position:
            candidates = day_data[day_data["EV"] >= ev_cut]

            if len(candidates) > 0:
                pick = candidates.sort_values("Max_Drawdown", ascending=False).iloc[0]

                price = pick["Close"]
                invest = daily_amount
                shares = invest / price

                total_shares = shares
                total_invested = invest
                entry_price = price
                holding_day = 1
                in_position = True
            else:
                idle_days += 1

        else:
            row = day_data[day_data["Ticker"] == pick["Ticker"]]
            if row.empty:
                continue

            row = row.iloc[0]
            holding_day += 1

            avg_price = total_invested / total_shares

            # ✅ 익절 (intraday)
            if row["High"] >= avg_price * 1.10:
                sell_price = avg_price * 1.10
                seed += total_shares * sell_price - total_invested
                in_position = False
                total_shares = 0
                total_invested = 0
                continue

            # ✅ 보유일 초과
            if holding_day >= max_days:
                if row["Low"] <= avg_price * (1 + stop_level):
                    sell_price = avg_price * (1 + stop_level)
                else:
                    sell_price = row["Close"]

                seed += total_shares * sell_price - total_invested
                in_position = False
                total_shares = 0
                total_invested = 0
                continue

            # ✅ 추가 매수 (종가 기준)
            close_price = row["Close"]

            if close_price <= avg_price * 1.05:
                if close_price >= avg_price:
                    invest = daily_amount * 0.5
                else:
                    invest = daily_amount

                shares = invest / close_price
                total_shares += shares
                total_invested += invest

        equity_curve.append(seed)

        if seed > max_equity:
            max_equity = seed

        dd = (seed - max_equity) / max_equity
        if dd < max_dd:
            max_dd = dd

    total_return = (seed / INITIAL_SEED) - 1

    return total_return, seed / INITIAL_SEED, max_dd, idle_days


# ========================
# 파라미터 루프
# ========================

for q in ev_quantiles:
    ev_cut = df["EV"].quantile(q)

    for max_days in holding_days_list:
        for stop_level in stop_levels:

            tr, multiple, mdd, idle = run_backtest(ev_cut, max_days, stop_level)

            results.append({
                "EV_quantile": q,
                "EV_cut": ev_cut,
                "Max_Holding_Days": max_days,
                "Stop_Level": stop_level,
                "Total_Return": tr,
                "Seed_Multiple": multiple,
                "Max_Drawdown": mdd,
                "Idle_Days": idle
            })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Seed_Multiple", ascending=False)

results_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Parametric backtest complete")
print(results_df.head(10))
