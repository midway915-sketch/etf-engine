import pandas as pd
import numpy as np

# ===============================
# 데이터 로드
# ===============================
df = pd.read_csv("data/raw_data.csv")

base_success_rate = df["Success_1st"].mean()
base_count = len(df)

print("\n📊 전체 데이터")
print("Samples:", base_count)
print("Base Success Rate:", round(base_success_rate, 4))

results = []

# ===============================
# 1️⃣ Drawdown_252 단독 필터
# ===============================
df_f1 = df[df["Drawdown_252"] < -0.25]

results.append({
    "Filter": "Drawdown_252 < -0.25",
    "Samples": len(df_f1),
    "Success_rate": df_f1["Success_1st"].mean()
})

# ===============================
# 2️⃣ Drawdown_252 + ATR_ratio
# ===============================
df_f2 = df[
    (df["Drawdown_252"] < -0.25) &
    (df["ATR_ratio"] > df["ATR_ratio"].median())
]

results.append({
    "Filter": "Drawdown_252 < -0.25 & ATR_ratio > median",
    "Samples": len(df_f2),
    "Success_rate": df_f2["Success_1st"].mean()
})

# ===============================
# 3️⃣ Drawdown_252 + ATR_ratio + Drawdown_60
# ===============================
df_f3 = df[
    (df["Drawdown_252"] < -0.25) &
    (df["ATR_ratio"] > df["ATR_ratio"].median()) &
    (df["Drawdown_60"] < -0.15)
]

results.append({
    "Filter": "DD252<-0.25 & ATR>med & DD60<-0.15",
    "Samples": len(df_f3),
    "Success_rate": df_f3["Success_1st"].mean()
})

# ===============================
# 결과 정리
# ===============================
result_df = pd.DataFrame(results)

result_df["Base_success_rate"] = base_success_rate
result_df["Improvement"] = result_df["Success_rate"] - base_success_rate

print("\n🔥 필터 실험 결과\n")
print(result_df)

# ===============================
# 저장
# ===============================
result_df.to_csv("data/filter_experiment_results.csv", index=False)

print("\n✅ 저장 완료 → data/filter_experiment_results.csv")
