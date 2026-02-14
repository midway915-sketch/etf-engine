import pandas as pd
import numpy as np
import joblib

# ===============================
# 설정
# ===============================
DATA_PATH = "data/raw_data.csv"
MODEL_PATH = "app/model.pkl"
SCALER_PATH = "app/scaler.pkl"

FEATURES = [
    "Drawdown_252",
    "Drawdown_60",
    "ATR_ratio",
    "Z_score",
    "MACD_hist",
    "MA20_slope",
    "Market_Drawdown",
    "Market_ATR_ratio"
]

# ===============================
# 데이터 로드
# ===============================
df = pd.read_csv(DATA_PATH)
df = df.sort_values("Date").reset_index(drop=True)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===============================
# 확률 예측
# ===============================
X = df[FEATURES]
X_scaled = scaler.transform(X)
df["Pred_Prob"] = model.predict_proba(X_scaled)[:, 1]

# ===============================
# 🔥 실패 평균 수익 계산 (성공 0.10 제외)
# ===============================
fail_mean = df.loc[
    df["Return_Fail2"] != 0.10,
    "Return_Fail2"
].mean()

# ===============================
# 🔥 올바른 EV 계산
# ===============================
df["EV"] = (
    df["Pred_Prob"] * 0.10
    + (1 - df["Pred_Prob"]) * fail_mean
)

# ===============================
# 🔥 테스트 구간 분리 (80% / 20%)
# ===============================
split_idx = int(len(df) * 0.8)
df_test = df.iloc[split_idx:].copy()

# ===============================
# 🔥 테스트 구간 통계 출력
# ===============================
df_ev_positive = df_test[df_test["EV"] > 0]
threshold = df_test["EV"].quantile(0.8)
df_top20 = df_test[df_test["EV"] >= threshold]

print("=" * 60)
print("📊 [TEST 구간 결과 - Corrected EV]")
print("전체 평균 EV:", round(df_test["EV"].mean(), 4))
print("EV > 0 비율:", round(len(df_ev_positive) / len(df_test), 4))
print("EV > 0 실제 성공률:", round(df_ev_positive["Success"].mean(), 4))
print("상위 20% 표본 개수:", len(df_top20))
print("상위 20% 실제 성공률:", round(df_top20["Success"].mean(), 4))

print("-" * 60)
print("Success 비율:", round(df_test["Success"].mean(), 4))
print("Fail 평균 수익:", round(fail_mean, 4))
print("Fail 최소:", round(
    df_test.loc[df_test["Return_Fail2"] != 0.10, "Return_Fail2"].min(), 4
))

print("-" * 60)
print("확률 평균:", round(df_test["Pred_Prob"].mean(), 4))
print("확률 최소:", round(df_test["Pred_Prob"].min(), 4))
print("확률 최대:", round(df_test["Pred_Prob"].max(), 4))
print("=" * 60)

# ===============================
# 저장 (테스트 결과만 저장)
# ===============================
df_test.to_csv("data/ev_results_test.csv", index=False)
print("✅ ev_results_test.csv 저장 완료")
