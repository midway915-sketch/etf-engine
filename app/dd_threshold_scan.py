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

if "Success" not in df.columns or "Max_Drawdown" not in df.columns:
    raise ValueError("필요한 컬럼이 없습니다.")

print("=" * 70)
print("📊 Max_Drawdown Threshold Scan (-10% ~ -20%)")
print("=" * 70)
print(f"{'Threshold':>12} | {'Count':>8} | {'Success Rate':>12}")
print("-" * 70)

# ===============================
# -10% ~ -20% (1% 단위)
# ===============================
for dd in np.arange(-0.10, -0.201, -0.01):
    filtered = df[df["Max_Drawdown"] > dd]

    count = len(filtered)

    if count > 0:
        success_rate = filtered["Success"].mean()
        print(f"{dd:>12.2%} | {count:>8} | {success_rate:>12.4f}")
    else:
        print(f"{dd:>12.2%} | {count:>8} | {'N/A':>12}")

print("=" * 70)
