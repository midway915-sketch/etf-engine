import pandas as pd
import numpy as np

# ===============================
# 파일 경로
# ===============================
DATA_PATH = "data/ev_top20_test_raw.csv"

# ===============================
# 데이터 로드
# ===============================
df = pd.read_csv(DATA_PATH)

# Success 컬럼이 반드시 있어야 함
if "Success" not in df.columns:
    raise ValueError("Success 컬럼이 없습니다.")

# ===============================
# 분석 대상 컬럼
# ===============================
feature_cols = [
    "Holding_Period",
    "Max_Drawdown",
    "Drawdown_60",
    "Drawdown_252",
    "Z_score",
    "ATR_ratio",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio",
    "Market_above_MA200",
    "Pred_Prob",
    "EV"
]

# 실제 존재하는 컬럼만 사용
feature_cols = [col for col in feature_cols if col in df.columns]

print("=" * 60)
print("📊 상관관계 분석 (Success vs 변수)")
print("=" * 60)

# ===============================
# 1️⃣ 피어슨 상관계수
# ===============================
corr_results = {}

for col in feature_cols:
    corr = df["Success"].corr(df[col])
    corr_results[col] = corr

corr_df = pd.DataFrame.from_dict(
    corr_results,
    orient="index",
    columns=["Correlation_with_Success"]
).sort_values(by="Correlation_with_Success", ascending=False)

print(corr_df)
print("-" * 60)

# ===============================
# 2️⃣ 성공/실패 그룹 평균 비교
# ===============================
print("📊 성공 vs 실패 평균 비교")
print("-" * 60)

group_mean = df.groupby("Success")[feature_cols].mean().T
group_mean.columns = ["Fail_mean", "Success_mean"]

group_mean["Diff(Success-Fail)"] = (
    group_mean["Success_mean"] - group_mean["Fail_mean"]
)

group_mean = group_mean.sort_values(
    by="Diff(Success-Fail)", ascending=False
)

print(group_mean)

print("=" * 60)
