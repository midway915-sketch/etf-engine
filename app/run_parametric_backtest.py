import pandas as pd
import numpy as np

INPUT_PATH = "data/backtest_panel.csv"
OUTPUT_PATH = "data/parametric_results_v2.csv"
INITIAL_SEED = 40_000_000

df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values(["Date", "Ticker"])

profit_targets = [0.09, 0.11, 0.13, 0.15]
ev_quantiles = [0.65, 0.68, 0.71, 0.74, 0.78]
holding_days_list = [20, 25, 30, 35, 40, 45]
stop_levels = [0.00, -0.03, -0.06, -0.09]

results = []

# ==========================================================
# 🔥 [수정 1] groupby 미리 생성 + list화 (속도 개선)
# ==========================================================
grouped = df.groupby("Date", sort=False)  # 🔥 수정
date_groups = list(grouped)               # 🔥 수정 (dict 대신 list)

# ==========================================================
# 🔥 [수정 2] EV quantile 미리 계산 (루프 밖)
# ==========================================================
ev_cut_map = {q: df["EV"].quantile(q) for q in ev_quantiles}  # 🔥 추가


def run_backtest(profit_target, ev_cut, max_days, stop_level, scenario):

    seed = INITIAL_SEED
    in_position = False
    idle_days = 0
    total_trades = 0
    win_trades = 0
    total_shares = 0
    total_invested = 0
    holding_day = 0
    extending = False
    actual_max_holding_days = 0
    max_equity = seed
    max_dd = 0

    # ==========================================================
    # 🔥 [수정 3] dict 조회 제거 → list unpack 방식
    # ==========================================================
    for date, day_data in date_groups:   # 🔥 수정
        daily_amount = seed / max_days

        if not in_position:
            candidates = day_data[day_data["EV"] >= ev_cut]
            if len(candidates) > 0:
                pick = candidates.sort_values("EV", ascending=False).iloc[0]
                price = pick["Close"]
                invest = daily_amount
                shares = invest / price

                total_shares = shares
                total_invested = invest
                seed -= invest
                holding_day = 1
                extending = False
                in_position = True
            else:
                idle_days += 1
        else:
            row = day_data[day_data["Ticker"] == pick["Ticker"]]
            if row.empty:
                continue

            row = row.iloc[0]
            holding_day += 1

            if holding_day > actual_max_holding_days:
                actual_max_holding_days = holding_day

            avg_price = total_invested / total_shares

            if row["High"] >= avg_price * (1 + profit_target):
                sell_price = avg_price * (1 + profit_target)
                proceeds = total_shares * sell_price
                profit = proceeds - total_invested

                seed += proceeds
                total_trades += 1
                if profit > 0:
                    win_trades += 1

                in_position = False
                total_shares = 0
                total_invested = 0
                continue

            if holding_day >= max_days and not extending:
                if scenario == 1:
                    sell_price = row["Close"]
                    proceeds = total_shares * sell_price
                    profit = proceeds - total_invested

                    seed += proceeds
                    total_trades += 1
                    if profit > 0:
                        win_trades += 1

                    in_position = False
                    total_shares = 0
                    total_invested = 0
                    continue
                else:
                    extending = True

            if extending:
                if row["Low"] <= avg_price * (1 + stop_level):
                    sell_price = avg_price * (1 + stop_level)
                    proceeds = total_shares * sell_price
                    profit = proceeds - total_invested

                    seed += proceeds
                    total_trades += 1
                    if profit > 0:
                        win_trades += 1

                    in_position = False
                    total_shares = 0
                    total_invested = 0
                    continue

            close_price = row["Close"]
            if close_price <= avg_price * 1.05:
                invest = daily_amount * 0.5 if close_price >= avg_price else daily_amount
                invest = min(invest, seed)

                if invest > 0:
                    shares = invest / close_price
                    total_shares += shares
                    total_invested += invest
                    seed -= invest

        # MDD 계산
        if in_position:
            row_current = day_data[day_data["Ticker"] == pick["Ticker"]]
            current_value = (
                total_shares * row_current.iloc[0]["Close"]
                if not row_current.empty
                else 0
            )
        else:
            current_value = 0

        equity = seed + current_value

        if equity > max_equity:
            max_equity = equity

        dd = (equity - max_equity) / max_equity
        if dd < max_dd:
            max_dd = dd

    total_return = (equity / INITIAL_SEED) - 1
    success_rate = win_trades / total_trades if total_trades > 0 else 0

    return total_return, equity / INITIAL_SEED, max_dd, idle_days, success_rate, total_trades, actual_max_holding_days


# ==========================================================
# 파라미터 루프
# ==========================================================

scenario = 2

for q in ev_quantiles:
    ev_cut = ev_cut_map[q]   # 🔥 수정 (미리 계산된 값 사용)

    for max_days in holding_days_list:
        for stop_level in stop_levels:
            for profit_target in profit_targets:

                tr, multiple, mdd, idle, sr, cycle_count, real_max_hold = run_backtest(
                    profit_target, ev_cut, max_days, stop_level, scenario
                )

                results.append({
                    "Scenario": scenario,
                    "EV_quantile": q,
                    "EV_cut": ev_cut,
                    "Profit_Target": profit_target,
                    "Max_Holding_Days": max_days,
                    "Actual_Max_Holding_Days": real_max_hold,
                    "Stop_Level": stop_level,
                    "Total_Return": tr,
                    "Seed_Multiple": multiple,
                    "Max_Drawdown": mdd,
                    "Idle_Days": idle,
                    "Success_Rate": sr,
                    "Cycle_Count": cycle_count
                })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Seed_Multiple", ascending=False)
results_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Parametric backtest v3 complete (Optimized loop)")
print(results_df.head(10))
