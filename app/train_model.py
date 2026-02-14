import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

df = pd.read_csv("data/success_dataset.csv")

X = df.drop(columns=["Date", "Ticker", "Close", "Success"])
y = df["Success"]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

model.fit(X, y)

proba = model.predict_proba(X)[:,1]
print("ROC-AUC:", roc_auc_score(y, proba))

joblib.dump(model, "models/success_model.pkl")
print("✅ 모델 저장 완료")
