import pandas as pd
import numpy as np

INPUT_PATH = "data/backtest_panel.csv"
OUTPUT_PATH = "data/parametric_results_v2.csv"
INITIAL_SEED = 40_000_000

df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values(["Date", "Ticker"])

ev_quantiles = [0.70, 0.75, 0.80, 0.85, 0.90]
holding_days_list = [30, 40, 50]
stop_levels = [-0.05, -0.10, -0.15]

results = []

# ==========================================================
# 백테스트 함수
# scenario = 1  → 보유일 도달 시 무조건 정리
# scenario = 2  → 보유일 이후에도 stop 도달까지 계속 매수
# 🔥 복리 구조 적용 (daily_amount = seed / max_days)
# 🔥 MDD = 총자산 기준 (현금 + 평가금)
# 🔥 Cycle_Count = 총 트레이드 횟수
# ==========================================================

def run_backtest(ev_cut, max_days, stop_level, scenario):

    seed = INITIAL_SEED
    in_position = False
    idle_days = 0
    total_trades = 0
    win_trades = 0
    total_shares = 0
    total_invested = 0
    holding_day = 0
    extending = False

    max_equity = seed
    max_dd = 0

    grouped = df.groupby("Date")

    for date, day_data in grouped:

        daily_amount = seed / max_days

        # ===============================
        # 신규 진입
        # ===============================
        if not in_position:

            candidates = day_data[day_data["EV"] >= ev_cut]

            if len(candidates) > 0:

                pick = candidates.sort_values("Max_Drawdown", ascending=False).iloc[0]

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

        # ===============================
        # 보유 중
        # ===============================
        else:

            row = day_data[day_data["Ticker"] == pick["Ticker"]]
            if row.empty:
                continue

            row = row.iloc[0]
            holding_day += 1

            avg_price = total_invested / total_shares

            # -----------------------------
            # 1️⃣ +10% 익절
            # -----------------------------
            if row["High"] >= avg_price * 1.10:

                sell_price = avg_price * 1.10
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

            # -----------------------------
            # 2️⃣ 보유일 도달
            # -----------------------------
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

                elif scenario == 2:
                    extending = True

            # -----------------------------
            # 3️⃣ Scenario 2 연장 구간
            # -----------------------------
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

            # -----------------------------
            # 4️⃣ 추가 매수
            # -----------------------------
            close_price = row["Close"]

            if close_price <= avg_price * 1.05:

                if close_price >= avg_price:
                    invest = daily_amount * 0.5
                else:
                    invest = daily_amount

                invest = min(invest, seed)

                if invest > 0:
                    shares = invest / close_price
                    total_shares += shares
                    total_invested += invest
                    seed -= invest

        # ===============================
        # 🔥 MDD 계산 (총자산 기준으로 변경)
        # ===============================
        if in_position:
            current_value = total_shares * row["Close"]
        else:
            current_value = 0

        equity = seed + current_value

        if equity > max_equity:
            max_equity = equity

        dd = (equity - max_equity) / max_equity

        if dd < max_dd:
            max_dd = dd

    # 🔥 최종 수익도 총자산 기준
    total_return = (equity / INITIAL_SEED) - 1
    success_rate = win_trades / total_trades if total_trades > 0 else 0

    return total_return, equity / INITIAL_SEED, max_dd, idle_days, success_rate, total_trades


# ==========================================================
# 파라미터 루프
# ==========================================================

for scenario in [1, 2]:
    for q in ev_quantiles:

        ev_cut = df["EV"].quantile(q)

        for max_days in holding_days_list:
            for stop_level in stop_levels:

                tr, multiple, mdd, idle, sr, cycle_count = run_backtest(
                    ev_cut, max_days, stop_level, scenario
                )

                results.append({
                    "Scenario": scenario,
                    "EV_quantile": q,
                    "EV_cut": ev_cut,
                    "Max_Holding_Days": max_days,
                    "Stop_Level": stop_level,
                    "Total_Return": tr,
                    "Seed_Multiple": multiple,
                    "Max_Drawdown": mdd,
                    "Idle_Days": idle,
                    "Success_Rate": sr,
                    "Cycle_Count": cycle_count  # 🔥 추가
                })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Seed_Multiple", ascending=False)
results_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Parametric backtest v3 complete (Real MDD + Cycle Count)")
print(results_df.head(10))
