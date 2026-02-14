import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 데이터 로드
df = pd.read_csv("data/success_dataset.csv")

# ===== Feature 생성 =====
def add_features(df):
    df = df.copy()

    df["Return_5"] = df.groupby("Ticker")["Close"].pct_change(5)
    df["Return_10"] = df.groupby("Ticker")["Close"].pct_change(10)
    df["Volatility_20"] = df.groupby("Ticker")["Close"].rolling(20).std().reset_index(0,drop=True)

    return df

df = add_features(df)
df = df.dropna()

FEATURES = [
    "Return_5",
    "Return_10",
    "Volatility_20"
]

X = df[FEATURES]
y = df["Target"]

# 학습 (고정)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

model.fit(X, y)

# 저장
joblib.dump(model, "model/model.pkl")

print("✅ model.pkl 저장 완료")

# 확률 저장
df["Probability"] = model.predict_proba(X)[:,1]
df.to_csv("data/scored_dataset.csv", index=False)

print("✅ scored_dataset.csv 생성 완료")
