import pandas as pd
import numpy as np

INPUT_PATH = "data/backtest_panel.csv"
OUTPUT_PATH = "data/parametric_results_v2.csv"
RAW_PATH = "data/cycle_raw_results.csv"

INITIAL_SEED = 40_000_000

df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values(["Date", "Ticker"])

profit_targets = [0.05, 0.10, 0.15]
ev_quantiles = [0.65, 0.70, 0.75]
holding_days_list = [20, 30, 40]
stop_levels = [0.00, -0.05, -0.10]

scenario = 2

# ============================================================
# 🔥 Numpy Engine (Max Loss 수정 + 결과 저장 정상화)
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

# 🔥 수정: 개별 사이클 최대 손실률 (사이클 내부 최저 수익률 추적)
cycle_min_return = np.zeros(P)
max_cycle_loss = np.zeros(P)   # 전체 사이클 중 가장 큰 손실률 저장

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

                # 🔥 수정: 사이클 시작 시 최소수익률 초기화
                cycle_min_return[i] = 0

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

            # 🔥 수정: 사이클 내부 현재 수익률 계산
            current_return = (row["Close"] - avg_price) / avg_price
            cycle_min_return[i] = min(cycle_min_return[i], current_return)

            # ---------- 익절 ----------
            if row["High"] >= avg_price * (1 + profit_target):
                sell_price = avg_price * (1 + profit_target)
                proceeds = total_shares[i] * sell_price
                cycle_return = (proceeds - total_invested[i]) / total_invested[i]

                # 🔥 수정: 최대 손실률 갱신
                max_cycle_loss[i] = min(max_cycle_loss[i], cycle_min_return[i])

                cycle_raw_records.append({
                    "Scenario": scenario,
                    "Param_Index": i,
                    "Ticker": picked_ticker[i],
                    "Start_Seed": cycle_start_seed[i],
                    "End_Seed": seed[i] + proceeds,
                    "Total_Invested": total_invested[i],
                    "Proceeds": proceeds,
                    "Cycle_Return": cycle_return,
                    "Min_Return_In_Cycle": cycle_min_return[i],  # 🔥 수정
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

                    # 🔥 수정: 최대 손실률 갱신
                    max_cycle_loss[i] = min(max_cycle_loss[i], cycle_min_return[i])

                    cycle_raw_records.append({
                        "Scenario": scenario,
                        "Param_Index": i,
                        "Ticker": picked_ticker[i],
                        "Start_Seed": cycle_start_seed[i],
                        "End_Seed": seed[i] + proceeds,
                        "Total_Invested": total_invested[i],
                        "Proceeds": proceeds,
                        "Cycle_Return": cycle_return,
                        "Min_Return_In_Cycle": cycle_min_return[i],  # 🔥 수정
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
# 🔥 Parametric 결과 정상 생성 (results 다시 생성)
# ============================================================

results = []  # 🔥 수정: results 정의 복구

for i, (q, ev_cut, profit_target, max_days, stop_level) in enumerate(param_grid):
    final_equity = seed[i]

    results.append({
        "Scenario": scenario,
        "EV_quantile": q,
        "EV_cut": ev_cut,
        "Profit_Target": profit_target,
        "Max_Holding_Days": max_days,
        "Stop_Level": stop_level,
        "Final_Seed": final_equity,
        "Seed_Multiple": final_equity / INITIAL_SEED,
        "Max_Loss_Rate": max_cycle_loss[i]   # 🔥 수정: 개별 사이클 최대손실률 반영
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Seed_Multiple", ascending=False)

# 🔥 수정: parametric 결과 저장 정상화
results_df.to_csv(OUTPUT_PATH, index=False)

# 🔥 수정: RAW 데이터 저장 정상화
cycle_raw_df = pd.DataFrame(cycle_raw_records)
cycle_raw_df.to_csv(RAW_PATH, index=False)

print("✅ Parametric result saved")
print(results_df.head())

print("✅ Cycle RAW data saved")
print(cycle_raw_df.head())
