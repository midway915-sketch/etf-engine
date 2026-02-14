import pandas as pd
import numpy as np
import joblib

# ===============================
# 설정
# ===============================
DATA_PATH = "data/raw_data.csv"
MODEL_PATH = "app/model.pkl"      # 🔥 수정
SCALER_PATH = "app/scaler.pkl"    # 🔥 수정


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
df = df.sort_values("Date")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===============================
# 확률 예측
# ===============================
X = df[FEATURES]
X_scaled = scaler.transform(X)

df["Pred_Prob"] = model.predict_proba(X_scaled)[:, 1]

# ===============================
# 🔥 기대값 계산 (Fail2 기준)
# ===============================
df["EV"] = (
    df["Pred_Prob"] * 0.10
    + (1 - df["Pred_Prob"]) * df["Return_Fail2"]
)

# ===============================
# 🔥 필터 예시
# ===============================

# 1️⃣ EV > 0
df_ev_positive = df[df["EV"] > 0]

# 2️⃣ EV 상위 20%
threshold = df["EV"].quantile(0.8)
df_top20 = df[df["EV"] >= threshold]

# ===============================
# 결과 출력
# ===============================
print("=" * 60)
print("전체 평균 EV:", round(df["EV"].mean(), 4))
print("EV > 0 비율:", round(len(df_ev_positive) / len(df), 4))
print("EV > 0 실제 성공률:", round(df_ev_positive["Success"].mean(), 4))
print("상위 20% 실제 성공률:", round(df_top20["Success"].mean(), 4))
print("=" * 60)

# 저장
df.to_csv("data/ev_results.csv", index=False)

print("✅ ev_results.csv 저장 완료")
