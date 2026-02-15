import pandas as pd
import numpy as np

# ============================================================
# 설정
# ============================================================
INPUT_PATH = "data/backtest_panel.csv"
PARAM_PATH = "data/parametric_results_v2.csv"
OUTPUT_PATH = "data/single_param_cycle_results.csv"

INITIAL_SEED = 40_000_000

# ============================================================
# 1️⃣ 최상위 파라미터 1개 선택
# ============================================================
param_df = pd.read_csv(PARAM_PATH)
param_df = param_df.sort_values("Seed_Multiple", ascending=False)

top = param_df.iloc[0]

q = top["EV_quantile"]
profit_target = top["Profit_Target"]
max_days = top["Max_Holding_Days"]
stop_level = top["Stop_Level"]

print("✅ Selected Parameter")
print(top)

# ============================================================
# 2️⃣ 데이터 로드
# ============================================================
df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values(["Date", "Ticker"])

ev_cut = df["EV"].quantile(q)

# ============================================================
# 3️⃣ 사이클 재계산 (RAW 파일 없이 직접 계산)
# ============================================================
seed = INITIAL_SEED
in_position = False

total_shares = 0
total_invested = 0
holding_day = 0
extending = False
picked_ticker = None

cycle_start_seed = 0
cycle_start_date = None

cycles = []

grouped = df.groupby("Date", sort=False)

for date, day_data in grouped:

    day_data = day_data.set_index("Ticker")

    # =========================
    # 진입 전
    # =========================
    if not in_position:
        candidates = day_data[day_data["EV"] >= ev_cut]
        if len(candidates) > 0:
            pick = candidates.sort_values("EV", ascending=False).iloc[0]
            ticker = pick.name
            price = pick["Close"]

            cycle_start_seed = seed
            invest = seed / max_days
            shares = invest / price

            total_shares = shares
            total_invested = invest
            seed -= invest

            holding_day = 1
            extending = False
            in_position = True
            picked_ticker = ticker
            cycle_start_date = date

    # =========================
    # 보유 중
    # =========================
    else:
        if picked_ticker not in day_data.index:
            continue

        row = day_data.loc[picked_ticker]
        holding_day += 1

        avg_price = total_invested / total_shares

        # ---------- 익절 ----------
        if row["High"] >= avg_price * (1 + profit_target):

            sell_price = avg_price * (1 + profit_target)
            proceeds = total_shares * sell_price

            cycle_return = (proceeds - total_invested) / total_invested

            cycles.append({
                "Start_Date": cycle_start_date,
                "End_Date": date,
                "Ticker": picked_ticker,
                "Holding_Days": holding_day,
                "Start_Seed": cycle_start_seed,
                "End_Seed": seed + proceeds,
                "Cycle_Return": cycle_return,
                "Exit_Type": "PROFIT"
            })

            seed += proceeds
            in_position = False

        # ---------- 연장 시작 ----------
        elif holding_day >= max_days:
            extending = True

        # ---------- 연장 손절 ----------
        if extending and row["Low"] <= avg_price * (1 + stop_level):

            sell_price = avg_price * (1 + stop_level)
            proceeds = total_shares * sell_price

            cycle_return = (proceeds - total_invested) / total_invested

            cycles.append({
                "Start_Date": cycle_start_date,
                "End_Date": date,
                "Ticker": picked_ticker,
                "Holding_Days": holding_day,
                "Start_Seed": cycle_start_seed,
                "End_Seed": seed + proceeds,
                "Cycle_Return": cycle_return,
                "Exit_Type": "STOP"
            })

            seed += proceeds
            in_position = False

        # ---------- 추가매수 ----------
        elif row["Close"] <= avg_price * 1.05:
            invest = cycle_start_seed / max_days
            shares = invest / row["Close"]
            total_shares += shares
            total_invested += invest
            seed -= invest

# ============================================================
# 4️⃣ 저장
# ============================================================
cycles_df = pd.DataFrame(cycles)

print("✅ Total Cycles:", len(cycles_df))
print(cycles_df.head())

cycles_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Saved:", OUTPUT_PATH)
