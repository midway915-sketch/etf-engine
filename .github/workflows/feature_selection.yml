import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ===============================
# 데이터 로드
# ===============================
df = pd.read_csv("data/raw_data.csv")

# ===============================
# 타겟 선택
# 1차 성공 확률 모델
# ===============================
y = df["Success_1st"]

# 전략 결과 컬럼 제거
drop_cols = [
    "Ticker",
    "Success_1st",
    "Return_1st",
    "Hold_days_1st",
    "Second_phase_used",
    "Return_final",
    "Hold_days_final",
    "Max_hold_days",
    "Max_drawdown"
]

X = df.drop(columns=drop_cols)

# ===============================
# 모델 학습
# ===============================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# ===============================
# 중요도 추출
# ===============================
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n🔥 Feature Importance")
print(importances.head(15))

# ===============================
# 상위 8~12개 선택
# ===============================
top_features = importances.head(12).index.tolist()

print("\n✅ 선택된 핵심 지표:")
for f in top_features:
    print(f)
