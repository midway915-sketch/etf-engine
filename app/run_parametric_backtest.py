import pandas as pd
import numpy as np

INPUT_PATH = "data/backtest_panel.csv"
OUTPUT_PATH = "data/parametric_results_v2.csv"
INITIAL_SEED = 40_000_000

df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values(["Date", "Ticker"])

profit_targets = [0.05, 0.10, 0.15]
ev_quantiles = [0.65, 0.70, 0.75]
holding_days_list = [35, 40, 45]
stop_levels = [-0.05, -0.08, -0.10, -0.12]
scenario = 2

# ============================================================
# 🔥 Numpy Engine (물타기 모드 반영)
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
picked_ticker = np.array([None] * P, dtype=object)
cycle_unit = np.zeros(P)
cycle_start_seed = np.zeros(P)
cycle_max_loss = np.zeros(P)

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
            else:
                idle_days[i] += 1

        # =========================
        # 보유 중
        # =========================
        else:
            if picked_ticker[i] not in day_data.index:
                continue

            row = day_data.loc[picked_ticker[i]]
            holding_day[i] += 1

            actual_max_holding_days[i] = max(
                actual_max_holding_days[i], holding_day[i]
            )

            avg_price = total_invested[i] / total_shares[i]

            # =====================================================
            # 🔥 max_days 도달 시 분기
            # =====================================================
            if holding_day[i] >= max_days and not extending[i]:
                current_return = (row["Close"] - avg_price) / avg_price

                if current_return >= stop_level:
                    sell_price = row["Close"]
                    proceeds = total_shares[i] * sell_price
                    cycle_return = (proceeds - total_invested[i]) / total_invested[i]

                    cycle_max_loss[i] = min(cycle_max_loss[i], cycle_return)

                    seed[i] += proceeds
                    total_trades[i] += 1
                    if cycle_return > 0:
                        win_trades[i] += 1

                    in_position[i] = False
                    total_shares[i] = 0
                    total_invested[i] = 0
                    holding_day[i] = 0
                    extending[i] = False
                    cycle_unit[i] = 0
                    cycle_start_seed[i] = 0
                    picked_ticker[i] = None
                    continue
                else:
                    extending[i] = True

            # =====================================================
            # 🔥 물타기 모드
            # =====================================================
            if extending[i]:
                # 🔥 stop_level 도달 시 청산
                if row["High"] >= avg_price * (1 + stop_level):
                    sell_price = avg_price * (1 + stop_level)
                    proceeds = total_shares[i] * sell_price
                    cycle_return = (proceeds - total_invested[i]) / total_invested[i]
                    cycle_max_loss[i] = min(cycle_max_loss[i], cycle_return)
                    seed[i] += proceeds
                    total_trades[i] += 1
                    in_position[i] = False
                    total_shares[i] = 0
                    total_invested[i] = 0
                    holding_day[i] = 0
                    extending[i] = False
                    cycle_unit[i] = 0
                    cycle_start_seed[i] = 0
                    picked_ticker[i] = None
                    continue
            
                # 🔥🔥🔥 수정: 물타기 모드에서는 조건 없이 매일 종가 full 매수
                if not daily_buy_done[i]:  # 🔥 수정
                    close_price = row["Close"]
                    invest = cycle_unit[i]  # 🔥 수정 (항상 full unit)
                    shares = invest / close_price
                    total_shares[i] += shares
                    total_invested[i] += invest
                    seed[i] -= invest
                    daily_buy_done[i] = True


            # =====================================================
            # 🔥 일반 구간 (DCA 조건부 매수로 수정된 부분)
            # =====================================================
            if not extending[i]:

                if row["High"] >= avg_price * (1 + profit_target):
                    sell_price = avg_price * (1 + profit_target)
                    proceeds = total_shares[i] * sell_price
                    cycle_return = (proceeds - total_invested[i]) / total_invested[i]

                    cycle_max_loss[i] = min(cycle_max_loss[i], cycle_return)

                    seed[i] += proceeds
                    total_trades[i] += 1
                    win_trades[i] += 1

                    in_position[i] = False
                    total_shares[i] = 0
                    total_invested[i] = 0
                    holding_day[i] = 0
                    extending[i] = False
                    cycle_unit[i] = 0
                    cycle_start_seed[i] = 0
                    picked_ticker[i] = None
                    continue

                # 🔥🔥🔥 수정된 추가매수 로직
                close_price = row["Close"]

                if not daily_buy_done[i]:

                    if close_price <= avg_price:
                        invest = cycle_unit[i]  # 🔥 평단 이하 → 전액 매수

                    elif close_price <= avg_price * 1.05:
                        invest = cycle_unit[i] / 2  # 🔥 평단 초과~5% → 반만 매수

                    else:
                        invest = 0  # 🔥 그 이상은 매수 안함

                    if invest > 0:
                        shares = invest / close_price
                        total_shares[i] += shares
                        total_invested[i] += invest
                        seed[i] -= invest
                        daily_buy_done[i] = True

        # =========================
        # MDD 계산 (기존 유지)
        # =========================
        if in_position[i] and picked_ticker[i] in day_data.index:
            current_price = day_data.loc[picked_ticker[i]]["Close"]
            current_value = total_shares[i] * current_price
        else:
            current_value = 0

        equity = seed[i] + current_value

        if equity > max_equity[i]:
            max_equity[i] = equity

        dd = (equity - max_equity[i]) / max_equity[i]

        if dd < max_dd[i]:
            max_dd[i] = dd

# ============================================================
# 결과 생성 (기존 그대로)
# ============================================================

results = []

for i, (q, ev_cut, profit_target, max_days, stop_level) in enumerate(param_grid):

    if in_position[i]:
        last_date = df["Date"].max()
        last_day = df[df["Date"] == last_date].set_index("Ticker")

        if picked_ticker[i] in last_day.index:
            current_value = total_shares[i] * last_day.loc[picked_ticker[i]]["Close"]
        else:
            current_value = 0
    else:
        current_value = 0

    final_equity = seed[i] + current_value

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
        "Total_Return": (final_equity / INITIAL_SEED) - 1,
        "Seed_Multiple": final_equity / INITIAL_SEED,
        "Max_Drawdown": max_dd[i],
        "Max_Loss_Rate": cycle_max_loss[i],
        "Idle_Days": idle_days[i],
        "Success_Rate": success_rate,
        "Cycle_Count": total_trades[i],
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Seed_Multiple", ascending=False)
results_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Numpy Engine Complete (Conditional DCA Applied)")
print(results_df.head(10))
