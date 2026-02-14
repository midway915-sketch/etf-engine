import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

# ===============================
# 설정
# ===============================
DATA_PATH = "data/raw_data.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

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

# 성공 정의
df["Target"] = (df["Return_final"] >= 0.10).astype(int)

df = df.dropna(subset=FEATURES + ["Target"])
df = df.sort_index()  # 시간순 정렬 (중요)

X = df[FEATURES]
y = df["Target"]

# ===============================
# 시계열 분할 (80% / 20%)
# ===============================
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ===============================
# 스케일링
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# Logistic + Isotonic Calibration
# ===============================
base_model = LogisticRegression(max_iter=300)

model = CalibratedClassifierCV(
    base_model,
    method="isotonic",
    cv=3
)

model.fit(X_train_scaled, y_train)

# ===============================
# 성능 확인
# ===============================
probs = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, probs)

print("=" * 50)
print(f"Test ROC-AUC: {auc:.4f}")
print("Base Success Rate:", y_test.mean())
print("Predicted Mean Probability:", probs.mean())
print("=" * 50)

# ===============================
# 저장
# ===============================
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("✅ model.pkl / scaler.pkl 생성 완료")
