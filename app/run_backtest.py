import pandas as pd
import numpy as np

# ==============================
# 설정값
# ==============================
INITIAL_CAPITAL = 40_000_000
DAILY_BUY = 1_000_000
EV_THRESHOLD = 0.05974

DATA_PATH = "data/backtest_panel.csv"
OUTPUT_PATH = "data/backtest_result_summary.csv"

# ==============================
# 로드
# ==============================
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values(["Date", "Ticker"])

dates = sorted(df["Date"].unique())

# ==============================
# 백테스트 함수
# ==============================
def run_backtest(maxdd_filter=None, scenario=1):

    capital = INITIAL_CAPITAL
    equity_curve = []
    in_position = False
    ticker = None
    avg_price = 0
    total_qty = 0
    hold_days = 0
    idle_days = 0
    max_hold = 0
    wins = 0
    trades = 0

    for i in range(1, len(dates)):
        today = dates[i]
        yesterday = dates[i-1]

        day_data = df[df["Date"] == today]
        prev_data = df[df["Date"] == yesterday]

        # ======================
        # 포지션 없을 때 진입
        # ======================
        if not in_position:
            candidates = prev_data[prev_data["EV"] >= EV_THRESHOLD]

            if maxdd_filter is not None:
                candidates = candidates[candidates["Max_Drawdown"] > maxdd_filter]

            if not candidates.empty:
                # 0에 더 가까운 Max_Drawdown 선택
                candidates["abs_dd"] = candidates["Max_Drawdown"].abs()
                selected = candidates.sort_values("abs_dd").iloc[0]
                ticker = selected["Ticker"]

                price = day_data[day_data["Ticker"] == ticker]["Close"].values
                if len(price) == 0:
                    idle_days += 1
                    equity_curve.append(capital)
                    continue

                price = price[0]
                qty = DAILY_BUY / price

                capital -= DAILY_BUY
                total_qty = qty
                avg_price = price
                in_position = True
                hold_days = 1
                trades += 1

            else:
                idle_days += 1
                equity_curve.append(capital)
                continue

        # ======================
        # 포지션 있을 때
        # ======================
        else:
            row = day_data[day_data["Ticker"] == ticker]
            if row.empty:
                equity_curve.append(capital)
                continue

            high = row["High"].values[0]
            low = row["Low"].values[0]
            close = row["Close"].values[0]

            # 🎯 +10% intraday 매도
            if high >= avg_price * 1.10:
                capital += total_qty * avg_price * 1.10
                wins += 1
                in_position = False
                total_qty = 0
                avg_price = 0
                max_hold = max(max_hold, hold_days)
                equity_curve.append(capital)
                continue

            hold_days += 1

            # 🔴 40일 초과 처리
            if hold_days > 40:
                if scenario == 1:
                    capital += total_qty * close
                    in_position = False
                    total_qty = 0
                    avg_price = 0
                    max_hold = max(max_hold, hold_days)
                    equity_curve.append(capital)
                    continue
                else:
                    # -10% intraday 도달 시 청산
                    if low <= avg_price * 0.90:
                        capital += total_qty * avg_price * 0.90
                        in_position = False
                        total_qty = 0
                        avg_price = 0
                        max_hold = max(max_hold, hold_days)
                        equity_curve.append(capital)
                        continue

            # 🟢 분할 매수 (종가 기준)
            if hold_days <= 40 or scenario == 2:
                if close <= avg_price:
                    buy_amount = DAILY_BUY
                elif close <= avg_price * 1.05:
                    buy_amount = DAILY_BUY / 2
                else:
                    buy_amount = 0

                if buy_amount > 0:
                    qty = buy_amount / close
                    capital -= buy_amount
                    total_qty += qty
                    avg_price = (
                        (avg_price * (total_qty - qty)) + (close * qty)
                    ) / total_qty

            equity_curve.append(capital + total_qty * close)

    equity_curve = pd.Series(equity_curve)
    total_return = equity_curve.iloc[-1] / INITIAL_CAPITAL - 1
    mdd = (equity_curve / equity_curve.cummax() - 1).min()
    success_rate = wins / trades if trades > 0 else 0

    return {
        "Scenario": scenario,
        "MaxDD_Filter": maxdd_filter,
        "Total_Return": round(total_return, 4),
        "Seed_Multiple": round(equity_curve.iloc[-1] / INITIAL_CAPITAL, 3),
        "Max_Drawdown": round(mdd, 4),
        "Max_Holding_Days": max_hold,
        "Success_Rate": round(success_rate, 4),
        "Idle_Days": idle_days,
    }

# ==============================
# 8가지 케이스 실행
# ==============================
filters = [None, -0.20, -0.15, -0.10]
results = []

for f in filters:
    results.append(run_backtest(f, scenario=1))
    results.append(run_backtest(f, scenario=2))

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_PATH, index=False)

print("✅ backtest_result_summary.csv 저장 완료")
print(result_df)
