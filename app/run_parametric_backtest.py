import pandas as pd
import numpy as np

INPUT_PATH = "data/backtest_panel.csv"
OUTPUT_PATH = "data/parametric_results_v2.csv"
INITIAL_SEED = 40_000_000

df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values(["Date", "Ticker"])

profit_targets = [0.09, 0.12, 0.15]
ev_quantiles = [0.65, 0.71, 0.78]
holding_days_list = [20, 30, 40]
stop_levels = [0.00, -0.10

scenario = 2

# -------------------------------
# 🔥 파라미터 조합 생성
# -------------------------------
param_grid = []
for q in ev_quantiles:
    ev_cut = df["EV"].quantile(q)
    for max_days in holding_days_list:
        for stop_level in stop_levels:
            for profit_target in profit_targets:
                param_grid.append((q, ev_cut, profit_target, max_days, stop_level))

P = len(param_grid)

# -------------------------------
# 🔥 상태 배열 초기화
# -------------------------------
seed = np.full(P, INITIAL_SEED, dtype=np.float64)
in_position = np.zeros(P, dtype=bool)
idle_days = np.zeros(P)
total_trades = np.zeros(P)
win_trades = np.zeros(P)
total_shares = np.zeros(P)
total_invested = np.zeros(P)
holding_day = np.zeros(P)
extending = np.zeros(P, dtype=bool)
actual_max_holding_days = np.zeros(P)
max_equity = np.full(P, INITIAL_SEED)
max_dd = np.zeros(P)

picked_ticker = np.array([None] * P, dtype=object)  # 🔥 수정 (파라미터별 ticker 상태 분리)

grouped = df.groupby("Date", sort=False)

# -------------------------------
# 🔥 날짜 루프 단 1번
# -------------------------------
for date, day_data in grouped:

    day_data = day_data.set_index("Ticker")

    for i, (q, ev_cut, profit_target, max_days, stop_level) in enumerate(param_grid):

        daily_amount = seed[i] / max_days

        # -------------------------------
        # 진입 전
        # -------------------------------
        if not in_position[i]:

            candidates = day_data[day_data["EV"] >= ev_cut]

            if len(candidates) > 0:
                pick = candidates.sort_values("EV", ascending=False).iloc[0]

                ticker = pick.name
                price = pick["Close"]

                invest = daily_amount
                shares = invest / price

                total_shares[i] = shares
                total_invested[i] = invest
                seed[i] -= invest

                holding_day[i] = 1
                extending[i] = False
                in_position[i] = True

                picked_ticker[i] = ticker  # 🔥 수정

            else:
                idle_days[i] += 1

        # -------------------------------
        # 보유 중
        # -------------------------------
        else:

            if picked_ticker[i] not in day_data.index:  # 🔥 수정
                continue

            row = day_data.loc[picked_ticker[i]]  # 🔥 수정

            holding_day[i] += 1
            actual_max_holding_days[i] = max(
                actual_max_holding_days[i], holding_day[i]
            )

            avg_price = total_invested[i] / total_shares[i]

            # 익절
            if row["High"] >= avg_price * (1 + profit_target):
                sell_price = avg_price * (1 + profit_target)
                proceeds = total_shares[i] * sell_price
                profit = proceeds - total_invested[i]

                seed[i] += proceeds
                total_trades[i] += 1

                if profit > 0:
                    win_trades[i] += 1

                in_position[i] = False
                total_shares[i] = 0
                total_invested[i] = 0

                continue

            # 보유일 도달
            if holding_day[i] >= max_days and not extending[i]:
                extending[i] = True

            # 손절 (연장 상태)
            if extending[i]:
                if row["Low"] <= avg_price * (1 + stop_level):

                    sell_price = avg_price * (1 + stop_level)
                    proceeds = total_shares[i] * sell_price
                    profit = proceeds - total_invested[i]

                    seed[i] += proceeds
                    total_trades[i] += 1

                    if profit > 0:
                        win_trades[i] += 1

                    in_position[i] = False
                    total_shares[i] = 0
                    total_invested[i] = 0

                    continue

            # 추가매수
            close_price = row["Close"]

            if close_price <= avg_price * 1.05:

                invest = daily_amount * 0.5 if close_price >= avg_price else daily_amount
                invest = min(invest, seed[i])

                if invest > 0:
                    shares = invest / close_price
                    total_shares[i] += shares
                    total_invested[i] += invest
                    seed[i] -= invest

        # -------------------------------
        # MDD 계산
        # -------------------------------
        if in_position[i] and picked_ticker[i] in day_data.index:  # 🔥 수정
            current_price = day_data.loc[picked_ticker[i]]["Close"]  # 🔥 수정
            current_value = total_shares[i] * current_price
        else:
            current_value = 0

        equity = seed[i] + current_value

        if equity > max_equity[i]:
            max_equity[i] = equity

        dd = (equity - max_equity[i]) / max_equity[i]

        if dd < max_dd[i]:
            max_dd[i] = dd

# -------------------------------
# 결과 생성
# -------------------------------
results = []

for i, (q, ev_cut, profit_target, max_days, stop_level) in enumerate(param_grid):

    equity = seed[i]

    total_return = (equity / INITIAL_SEED) - 1

    success_rate = (
        win_trades[i] / total_trades[i] if total_trades[i] > 0 else 0
    )

    results.append({
        "Scenario": scenario,
        "EV_quantile": q,
        "EV_cut": ev_cut,
        "Profit_Target": profit_target,
        "Max_Holding_Days": max_days,
        "Actual_Max_Holding_Days": actual_max_holding_days[i],
        "Stop_Level": stop_level,
        "Total_Return": total_return,
        "Seed_Multiple": equity / INITIAL_SEED,
        "Max_Drawdown": max_dd[i],
        "Idle_Days": idle_days[i],
        "Success_Rate": success_rate,
        "Cycle_Count": total_trades[i]
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Seed_Multiple", ascending=False)
results_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Numpy Engine Complete (Single Date Loop)")
print(results_df.head(10))
