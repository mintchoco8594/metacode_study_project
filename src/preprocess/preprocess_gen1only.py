import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]   # src
DATA_DIR = BASE_DIR / "data"

p1 = DATA_DIR / "Pokemon1.csv"
p2 = DATA_DIR / "pokemon2.csv"

# --- 1) Pokemon1.csv 읽기 (UTF-16 + 첫 줄 "#,Name..." 주석 처리)
df1 = pd.read_csv(
    p1
)
df1.columns = [c.strip() for c in df1.columns]

# "#"(도감번호) 숫자화 + 151까지 + Mega 제거
df1["#"] = pd.to_numeric(df1["#"], errors="coerce")
df1 = df1[(df1["#"] <= 151) & (~df1["Name"].astype(str).str.contains("Mega", na=False))].copy()

# --- 2) Pokemon2.csv 읽기 (탭 구분)
df2 = pd.read_csv(
    p2,
    encoding="utf-16",
    sep="\t"
)
df2.columns = [c.strip() for c in df2.columns]

# national_number 숫자화 + 151까지
df2["national_number"] = pd.to_numeric(df2["national_number"], errors="coerce")
df2 = df2[df2["national_number"] <= 151].copy()

# --- 3) 병합 키 만들기 (번호 기준이 제일 안전)
# df1: "#" / df2: "national_number"
df_merged = df1.merge(
    df2,
    left_on="#",
    right_on="national_number",
    how="inner",
    suffixes=("_p1", "_p2")
)

# --- 4) (선택) 이름 불일치 검증용 (영문명 기준)
# 포켓몬1 Name = "Charizard" / 포켓몬2 english_name = "Charizard"
# 만약 혹시라도 깨지면 아래로 확인 가능
# mismatch = df_merged[df_merged["Name"].str.strip().str.lower() != df_merged["english_name"].str.strip().str.lower()]
# print("name mismatch:", len(mismatch))

# --- 5) 저장
out = BASE_DIR.parent / "pokemon_gen1_merged.csv"
df_merged.to_csv(out, index=False, encoding="utf-8-sig")

print("pokemon1 filtered:", len(df1))
print("pokemon2 filtered:", len(df2))
print("merged:", len(df_merged))
print("saved:", out)
