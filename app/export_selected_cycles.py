import pandas as pd

# ============================================================
# 설정값 (여기만 바꿔서 쓰면 됨)
# ============================================================

PARAMETRIC_PATH = "data/parametric_results_v2.csv"
RAW_PATH = "data/cycle_raw_results.csv"
OUTPUT_PATH = "data/selected_193_cycles.csv"

# 👇 보고 싶은 파라미터 조건 (여기 맞춰서 수정 가능)
TARGET_SCENARIO = 2
TARGET_EV_Q = 0.65
TARGET_PROFIT_TARGET = 0.15
TARGET_MAX_HOLD = 20
TARGET_STOP = 0.0

# ============================================================
# 데이터 로드
# ============================================================

param_df = pd.read_csv(PARAMETRIC_PATH)
raw_df = pd.read_csv(RAW_PATH)

# ============================================================
# 1️⃣ parametric 결과에서 해당 row 찾기
# ============================================================

target_row = param_df[
    (param_df["Scenario"] == TARGET_SCENARIO) &
    (param_df["EV_quantile"] == TARGET_EV_Q) &
    (param_df["Profit_Target"] == TARGET_PROFIT_TARGET) &
    (param_df["Max_Holding_Days"] == TARGET_MAX_HOLD) &
    (param_df["Stop_Level"] == TARGET_STOP)
]

if len(target_row) == 0:
    raise ValueError("❌ 해당 조건의 파라미터가 없습니다.")

print("✅ Target Param Found")
print(target_row)

# ============================================================
# 2️⃣ Param_Index 찾기
# ============================================================

# parametric 파일에 Param_Index가 없다면
# raw 파일에서 조건으로 직접 필터링
selected_cycles = raw_df[
    (raw_df["Scenario"] == TARGET_SCENARIO) &
    (raw_df["EV_quantile"] == TARGET_EV_Q) &
    (raw_df["Profit_Target"] == TARGET_PROFIT_TARGET) &
    (raw_df["Max_Holding_Days"] == TARGET_MAX_HOLD) &
    (raw_df["Stop_Level"] == TARGET_STOP)
]

print(f"✅ Selected Cycle Count: {len(selected_cycles)}")

# ============================================================
# 3️⃣ 저장
# ============================================================

selected_cycles.to_csv(OUTPUT_PATH, index=False)

print("✅ 193 Cycles Saved")
print(selected_cycles.head())
