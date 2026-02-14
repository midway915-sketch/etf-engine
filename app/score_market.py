import pandas as pd
import joblib

df = pd.read_csv("data/success_dataset.csv")
model = joblib.load("models/success_model.pkl")

X = df.drop(columns=["Date", "Ticker", "Close", "Success"])
df["Probability"] = model.predict_proba(X)[:,1]

df.to_csv("data/scored_dataset.csv", index=False)
print("✅ 점수 계산 완료")
