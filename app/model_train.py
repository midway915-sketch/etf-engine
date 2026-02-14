import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

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

# 🔥 날짜 기준 정렬 (시계열 누수 방지)
df = df.sort_values("Date")

# 🔥 타겟 정의 (40일 내 +10% 성공 여부)
df["Target"] = df["Success"]

df = df.dropna(subset=FEATURES + ["Target"])

X = df[FEATURES]
y = df["Target"]

# ===============================
# 시계열 분할 (80% / 20%)
# ===============================
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# ===============================
# 스케일링 (train 기준만 fit)
# ===============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# Logistic + Isotonic Calibration
# 🔥 TimeSeriesSplit 사용 (누수 차단)
# ===============================
base_model = LogisticRegression(max_iter=500)

tscv = TimeSeriesSplit(n_splits=3)

model = CalibratedClassifierCV(
    base_model,
    method="isotonic",
    cv=tscv
)

model.fit(X_train_scaled, y_train)

# ===============================
# 성능 확인
# ===============================
probs = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, probs)

print("=" * 60)
print("Test ROC-AUC:", round(auc, 4))
print("Base Success Rate:", round(y_test.mean(), 4))
print("Predicted Mean Probability:", round(probs.mean(), 4))
print("=" * 60)

# ===============================
# 저장
# ===============================
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("✅ model.pkl / scaler.pkl 생성 완료")
