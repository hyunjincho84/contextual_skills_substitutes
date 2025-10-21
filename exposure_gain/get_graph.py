# -*- coding: utf-8 -*-
"""
job posting 단위 margin 비율 시각화 (양수 margin만, x축 로그 비닝 + 막대 테두리 추가)
- 한 posting = (row_idx, file_path)
- margin_ratio = (subs_trend - truth_trend) / truth_trend
- posting 내 여러 row 중 margin_ratio의 최대값 사용
- margin_ratio > 0 비율/개수 출력
- x축만 log binning (10^-1 ~ max)
- 막대 테두리(edge) 및 linewidth 추가
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === 경로 설정 ===
IN_CSV = "./sample_1000_scored_with_trends.csv"
ENCODING = "utf-8"

# === 데이터 로드 ===
df = pd.read_csv(IN_CSV, encoding=ENCODING)

# === 필수 컬럼 확인 ===
required_cols = {"truth_trend", "subs_trend", "row_idx", "file_path"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"⚠️ 필수 컬럼 누락: {missing}")

# === NaN 및 0 division 방지 ===
df = df.dropna(subset=["truth_trend", "subs_trend"]).copy()
df = df[df["truth_trend"] != 0]

# === margin 비율 계산 (백분율 기준) ===
df["margin_ratio"] = (df["subs_trend"] - df["truth_trend"]) / df["truth_trend"] * 100

# === job posting 단위 최대 margin 비율 ===
group_cols = ["row_idx", "file_path"]
postings = (
    df.groupby(group_cols, as_index=False)["margin_ratio"]
      .max()
      .rename(columns={"margin_ratio": "margin_ratio_max"})
)

# === 통계 출력 ===
total_posts = len(postings)
positive_posts = (postings["margin_ratio_max"] > 0).sum()
ratio_posts = positive_posts / total_posts * 100 if total_posts > 0 else 0.0

print(f"✅ Job postings 기준 — 전체 {total_posts:,}개 중 margin_ratio > 0 인 경우: {positive_posts:,}개 ({ratio_posts:.2f}%)")

# === 양수만 필터링 ===
postings_pos = postings.loc[postings["margin_ratio_max"] > 0].copy()

# === 로그 bin 정의 (10^-1 ~ max)
min_val = 1e-1
max_val = postings_pos["margin_ratio_max"].max()
bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)

# === 시각화 ===
sns.set(style="ticks", font_scale=1.4)
plt.figure(figsize=(8, 5))

sns.histplot(
    postings_pos["margin_ratio_max"],
    bins=bins,
    color="#55C0C2",
    edgecolor="#333333",   # ✅ 진한 회색 테두리
    linewidth=0.6,         # ✅ 막대 경계선 두께
    alpha=0.85,
)

plt.xscale("log")  # ✅ x축만 로그 스케일
plt.margins(x=0)
plt.grid(False)

# === 라벨 및 폰트 ===
plt.xlabel("Exposure Gain (%)", fontsize=24, labelpad=10)
plt.ylabel("Count", fontsize=24, labelpad=10)

# === 여백 및 저장 ===
plt.tight_layout()
plt.savefig("./margin_ratio_distribution.png", dpi=300, bbox_inches="tight")