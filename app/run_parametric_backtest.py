import pandas as pd
import numpy as np

INPUT_PATH = "data/backtest_panel.csv"
OUTPUT_PATH = "data/cycle_raw_results.csv"   # 🔥 수정: raw 전용 파일로 변경
INITIAL_SEED = 40_000_000

df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values(["Date", "Ticker"])

profit_targets = [0.05, 0.10, 0.15]
ev_quantiles = [0.65, 0.70, 0.75]
holding_days_list = [20, 30, 40]
stop_levels = [0.00, -0.05, -0.10]

scenario = 2

# ============================================================
# 🔥 Numpy Engine (사이클 RAW 저장 전용)
# ============================================================

param_grid = []
for q in ev_quantiles:
    ev_cut = df["EV"].quantile(q)
    for max_days in holding_days_list:
        for stop_level in stop_levels:
            for profit_target in profit_targets:
                param_grid.append((q, ev_cut, profit_target, max_days, stop_level))

P = len(param_grid)

seed = np.full(P, INITIAL_SEED, dtype=np.float64)
in_position = np.zeros(P, dtype=bool)
total_shares = np.zeros(P)
total_invested = np.zeros(P)
holding_day = np.zeros(P)
extending = np.zeros(P, dtype=bool)
picked_ticker = np.array([None] * P, dtype=object)
cycle_unit = np.zeros(P)
cycle_start_seed = np.zeros(P)

cycle_max_loss = np.zeros(P)

# 🔥 수정: raw 저장용 리스트
cycle_raw_records = []

grouped = df.groupby("Date", sort=False)

for date, day_data in grouped:
    day_data = day_data.set_index("Ticker")
    daily_buy_done = np.zeros(P, dtype=bool)

    for i, (q, ev_cut, profit_target, max_days, stop_level) in enumerate(param_grid):

        # =========================
        # 진입 전
        # =========================
        if not in_position[i]:
            candidates = day_data[day_data["EV"] >= ev_cut]
            if len(candidates) > 0 and not daily_buy_done[i]:
                pick = candidates.sort_values("EV", ascending=False).iloc[0]
                ticker = pick.name
                price = pick["Close"]

                cycle_start_seed[i] = seed[i]
                cycle_unit[i] = cycle_start_seed[i] / max_days

                invest = cycle_unit[i]
                shares = invest / price

                total_shares[i] = shares
                total_invested[i] = invest
                seed[i] -= invest

                holding_day[i] = 1
                extending[i] = False
                in_position[i] = True
                picked_ticker[i] = ticker
                daily_buy_done[i] = True

        # =========================
        # 보유 중
        # =========================
        else:
            if picked_ticker[i] not in day_data.index:
                continue

            row = day_data.loc[picked_ticker[i]]
            holding_day[i] += 1
            avg_price = total_invested[i] / total_shares[i]

            # ---------- 익절 ----------
            if row["High"] >= avg_price * (1 + profit_target):
                sell_price = avg_price * (1 + profit_target)
                proceeds = total_shares[i] * sell_price

                cycle_return = (proceeds - total_invested[i]) / total_invested[i]
                cycle_max_loss[i] = min(cycle_max_loss[i], cycle_return)

                # 🔥 수정: raw 저장
                cycle_raw_records.append({
                    "Scenario": scenario,
                    "Param_Index": i,
                    "Ticker": picked_ticker[i],
                    "Start_Seed": cycle_start_seed[i],
                    "End_Seed": seed[i] + proceeds,
                    "Total_Invested": total_invested[i],
                    "Proceeds": proceeds,
                    "Cycle_Return": cycle_return,
                    "Holding_Days": holding_day[i],
                    "Exit_Type": "PROFIT"
                })

                seed[i] += proceeds

                in_position[i] = False
                total_shares[i] = 0
                total_invested[i] = 0
                holding_day[i] = 0
                extending[i] = False
                cycle_unit[i] = 0
                cycle_start_seed[i] = 0
                picked_ticker[i] = None

                continue

            # ---------- 연장 시작 ----------
            if holding_day[i] >= max_days and not extending[i]:
                extending[i] = True

            # ---------- 연장 손절 ----------
            if extending[i]:
                if row["Low"] <= avg_price * (1 + stop_level):

                    sell_price = avg_price * (1 + stop_level)
                    proceeds = total_shares[i] * sell_price

                    cycle_return = (proceeds - total_invested[i]) / total_invested[i]
                    cycle_max_loss[i] = min(cycle_max_loss[i], cycle_return)

                    # 🔥 수정: raw 저장
                    cycle_raw_records.append({
                        "Scenario": scenario,
                        "Param_Index": i,
                        "Ticker": picked_ticker[i],
                        "Start_Seed": cycle_start_seed[i],
                        "End_Seed": seed[i] + proceeds,
                        "Total_Invested": total_invested[i],
                        "Proceeds": proceeds,
                        "Cycle_Return": cycle_return,
                        "Holding_Days": holding_day[i],
                        "Exit_Type": "STOP"
                    })

                    seed[i] += proceeds

                    in_position[i] = False
                    total_shares[i] = 0
                    total_invested[i] = 0
                    holding_day[i] = 0
                    extending[i] = False
                    cycle_unit[i] = 0
                    cycle_start_seed[i] = 0
                    picked_ticker[i] = None

                    continue

            # ---------- 추가매수 ----------
            close_price = row["Close"]
            if close_price <= avg_price * 1.05 and not daily_buy_done[i]:
                invest = cycle_unit[i]
                shares = invest / close_price
                total_shares[i] += shares
                total_invested[i] += invest
                seed[i] -= invest
                daily_buy_done[i] = True


# ============================================================
# 🔥 수정: RAW CSV 저장
# ============================================================

cycle_raw_df = pd.DataFrame(cycle_raw_records)
cycle_raw_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Cycle RAW data saved")
print(cycle_raw_df.head())
