import pandas as pd
import numpy as np

INPUT_PATH = "data/backtest_panel.csv"
OUTPUT_PATH = "data/parametric_results_v2.csv"
INITIAL_SEED = 40_000_000

df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
df = df.sort_values(["Date", "Ticker"])

profit_targets = [0.05, 0.10, 0.15]
ev_quantiles = [0.65, 0.70, 0.75]
holding_days_list = [20, 30, 40]
stop_levels = [0.00, ,-0.05, -0.10]
scenario = 2

# ============================================================
# 🔥 원본 엔진 (전략 로직 그대로)
# ============================================================

grouped_original = list(df.groupby("Date", sort=False))

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

    for date, day_data in grouped_original:
        day_data = day_data.set_index("Ticker")
        daily_amount = seed / max_days

        if not in_position:
            candidates = day_data[day_data["EV"] >= ev_cut]
            if len(candidates) > 0:
                pick = candidates.sort_values("EV", ascending=False).iloc[0]
                ticker = pick.name
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
            if ticker not in day_data.index:
                continue

            row = day_data.loc[ticker]
            holding_day += 1
            actual_max_holding_days = max(actual_max_holding_days, holding_day)
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

        if in_position and ticker in day_data.index:
            current_price = day_data.loc[ticker]["Close"]
            current_value = total_shares * current_price
        else:
            current_value = 0

        equity = seed + current_value
        max_equity = max(max_equity, equity)
        dd = (equity - max_equity) / max_equity
        max_dd = min(max_dd, dd)

    total_return = (equity / INITIAL_SEED) - 1
    success_rate = win_trades / total_trades if total_trades > 0 else 0

    return (
        total_return,
        equity / INITIAL_SEED,
        max_dd,
        idle_days,
        success_rate,
        total_trades,
        actual_max_holding_days,
    )

# ============================================================
# 🔥 Numpy Engine
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

grouped = df.groupby("Date", sort=False)

for date, day_data in grouped:
    day_data = day_data.set_index("Ticker")

    for i, (q, ev_cut, profit_target, max_days, stop_level) in enumerate(param_grid):

        daily_amount = seed[i] / max_days

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
                picked_ticker[i] = ticker
            else:
                idle_days[i] += 1
        else:
            if picked_ticker[i] not in day_data.index:
                continue

            row = day_data.loc[picked_ticker[i]]
            holding_day[i] += 1
            actual_max_holding_days[i] = max(
                actual_max_holding_days[i], holding_day[i]
            )
            avg_price = total_invested[i] / total_shares[i]

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

            if holding_day[i] >= max_days and not extending[i]:
                extending[i] = True

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

            close_price = row["Close"]
            if close_price <= avg_price * 1.05:
                invest = daily_amount * 0.5 if close_price >= avg_price else daily_amount
                invest = min(invest, seed[i])
                if invest > 0:
                    shares = invest / close_price
                    total_shares[i] += shares
                    total_invested[i] += invest
                    seed[i] -= invest

        if in_position[i] and picked_ticker[i] in day_data.index:
            current_price = day_data.loc[picked_ticker[i]]["Close"]
            current_value = total_shares[i] * current_price
        else:
            current_value = 0

        equity = seed[i] + current_value
        max_equity[i] = max(max_equity[i], equity)
        dd = (equity - max_equity[i]) / max_equity[i]
        max_dd[i] = min(max_dd[i], dd)

# ============================================================
# 🔥 단일 파라미터 완전 일치 검증 (수정 부분)
# ============================================================

TEST_Q = 0.65
TEST_PROFIT = 0.15
TEST_MAX_DAYS = 20
TEST_STOP = 0.00

ev_cut = df["EV"].quantile(TEST_Q)

orig = run_backtest(
    TEST_PROFIT,
    ev_cut,
    TEST_MAX_DAYS,
    TEST_STOP,
    scenario
)

print("\n===== ORIGINAL ENGINE =====")
print(orig)

for i, (q, ev_cut, profit_target, max_days, stop_level) in enumerate(param_grid):
    if (
        q == TEST_Q
        and profit_target == TEST_PROFIT
        and max_days == TEST_MAX_DAYS
        and stop_level == TEST_STOP
    ):
        # 🔥 수정: 마지막 포지션 평가금액 반영
        if in_position[i]:
            last_date = df["Date"].max()
            last_day = df[df["Date"] == last_date].set_index("Ticker")
            if picked_ticker[i] in last_day.index:
                current_value = total_shares[i] * last_day.loc[picked_ticker[i]]["Close"]
            else:
                current_value = 0
        else:
            current_value = 0

        final_equity = seed[i] + current_value  # 🔥 수정

        numpy_result = (
            (final_equity / INITIAL_SEED) - 1,   # 🔥 수정
            final_equity / INITIAL_SEED,         # 🔥 수정
            max_dd[i],
            idle_days[i],
            win_trades[i] / total_trades[i] if total_trades[i] > 0 else 0,
            total_trades[i],
            actual_max_holding_days[i],
        )

        print("\n===== NUMPY ENGINE =====")
        print(numpy_result)
        break
