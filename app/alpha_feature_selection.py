import pandas as pd
import numpy as np

# ===============================
# 데이터 로드
# ===============================
df = pd.read_csv("data/raw_data.csv")

# ===============================
# 분석 대상 feature (RF 상위 12개)
# ===============================
top_features = [
    "ATR_ratio",
    "Realized_vol",
    "Drawdown_252",
    "Drawdown_60",
    "BB_width",
    "MA120_gap",
    "Market_ATR_ratio",
    "MA60_gap",
    "Market_Drawdown",
    "Z_score",
    "MACD_hist",
    "MA20_slope"
]

# ===============================
# 성공 / 실패 분리
# ===============================
success_df = df[df["Success_1st"] == 1]
fail_df = df[df["Success_1st"] == 0]

report_rows = []

for col in top_features:
    success_mean = success_df[col].mean()
    fail_mean = fail_df[col].mean()
    diff = success_mean - fail_mean
    
    # 성공률이 높은 구간 확인 (하위 40% vs 상위 40%)
    low_threshold = df[col].quantile(0.4)
    high_threshold = df[col].quantile(0.6)

    low_group = df[df[col] <= low_threshold]
    high_group = df[df[col] >= high_threshold]

    low_success_rate = low_group["Success_1st"].mean()
    high_success_rate = high_group["Success_1st"].mean()

    report_rows.append({
        "Feature": col,
        "Success_mean": success_mean,
        "Fail_mean": fail_mean,
        "Mean_diff": diff,
        "Low_40%_success_rate": low_success_rate,
        "High_40%_success_rate": high_success_rate,
        "Success_rate_diff": low_success_rate - high_success_rate
    })

report = pd.DataFrame(report_rows)
report = report.sort_values("Mean_diff", key=abs, ascending=False)

# ===============================
# 출력 + 저장
# ===============================
print("\n🔥 성공 vs 실패 비교 리포트\n")
print(report)

report.to_csv("data/success_fail_comparison.csv", index=False)
print("✅ success_fail_comparison.csv 저장 완료")
