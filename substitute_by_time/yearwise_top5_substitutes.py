# yearwise_top5_mostsubs_postnorm.py
# -*- coding: utf-8 -*-
"""
For a TARGET_SKILL, compute YEAR-wise Top-K substitutes using posting-level 'most-sub' weights.

핵심 아이디어
- 동일 포스팅(row_idx) 안에서 동일 truth가 n회 등장하면,
  각 occurrence에서 '최다(sub)' 1개(=기본 Top-1, 단 Top-1이 truth면 Top-2)를 선택.
- 포스팅 단위에서 후보별 등장 횟수를 합하고, 포스팅 내 총 occurrence n으로 나눠 1/n 비율로 분배(가중치).
- 이렇게 얻은 포스팅 단위 가중치를 연도별로 합산하여 최종 순위 산출.
- 최종 랭킹 기준은 연도별 weight 합계(weight_sum)이며, 보기용으로 연도 내 정규화(weight_norm)도 제공.

입력 포맷(각 with_rowidx CSV):
  columns: row_idx, truth, pred_top5, pred_top5_probs
  파일명에서 year 추출 (정규식 r"(20\\d{2})")
출력:
  truth, year, rank, substitute, weight_sum, weight_norm
"""

import os
import re
import glob
import math
import argparse
import pandas as pd
from tqdm import tqdm

# ------------------------
# Helpers
# ------------------------
def _safe_lower(s):
    return str(s).lower() if pd.notna(s) else ""

def _find_rowidx_files(pred_root: str):
    """pred_root/2010..2025/... 에서 *with_rowidx.csv 파일 수집"""
    files = []
    for year_dir in sorted(glob.glob(os.path.join(pred_root, "[0-9]"*4))):
        if not os.path.isdir(year_dir):
            continue
        cand = glob.glob(os.path.join(year_dir, "*with_rowidx.csv"))
        if cand:
            files.extend(sorted(cand))
    if not files:
        raise FileNotFoundError(f"No *with_rowidx*.csv files under {pred_root}")
    return files

def _pick_most_sub_for_row(pred_top5: str, pred_probs: str, truth_lower: str):
    """
    한 행(occurrence)에서 '최다(sub)' 후보 1개만 선택.
    - 기본: Top-1 후보 (pred_top5의 첫 항목)
    - 단, Top-1이 truth면 Top-2 사용
    - 둘 다 truth거나 후보가 없으면 None
    """
    cands = [c.strip() for c in str(pred_top5 or "").split("|") if c.strip()]

    # probs는 길이 불일치가 있을 수 있으므로 방어적으로 파싱만 (사용하지 않지만 데이터 무결성 보정용)
    probs = []
    for p in str(pred_probs or "").split("|"):
        try:
            v = float(p)
            if math.isfinite(v):
                probs.append(v)
        except:
            pass
    if probs and len(probs) != len(cands):
        m = min(len(cands), len(probs))
        cands, probs = cands[:m], probs[:m]

    if not cands:
        return None

    # Top-1 우선
    top1 = _safe_lower(cands[0])
    if top1 != truth_lower:
        return cands[0]

    # Top-1이 truth면 Top-2 시도
    if len(cands) >= 2 and _safe_lower(cands[1]) != truth_lower:
        return cands[1]

    # 후보가 모두 truth거나 적절치 않은 경우
    return None

# ------------------------
# Load
# ------------------------
def load_with_rowidx(pred_root: str) -> pd.DataFrame:
    files = _find_rowidx_files(pred_root)
    dfs = []
    for fp in tqdm(files, desc="Loading *with_rowidx* predictions"):
        try:
            df = pd.read_csv(
                fp,
                usecols=["row_idx", "truth", "pred_top5", "pred_top5_probs"]
            )
            # 파일명에서 year 추출
            m = re.search(r"(20\d{2})", os.path.basename(fp))
            df["year"] = m.group(1) if m else None
            df["__src"] = fp
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] skip {fp}: {e}")
    if not dfs:
        raise RuntimeError("No valid with_rowidx CSVs loaded.")
    return pd.concat(dfs, ignore_index=True)

# ------------------------
# Core
# ------------------------
def yearwise_topk_mostsubs_postnorm(
    df: pd.DataFrame,
    target_skill: str,
    out_topk: int = 5
) -> pd.DataFrame:
    """
    포스팅(row_idx) 단위에서 most-sub 1개만 선택하여,
    포스팅 내 등장횟수 비율(=1/n)로 가중치 부여 후,
    연도별로 합산하여 최종 Top-K 랭킹.

    Returns columns:
      ["truth","year","rank","substitute","weight_sum","weight_norm"]
    """
    t_low = _safe_lower(target_skill)
    mask = df["truth"].astype(str).str.lower() == t_low
    df_t = df.loc[mask].copy()
    if df_t.empty:
        raise ValueError(f"No rows for TARGET_SKILL='{target_skill}'")

    # 1) 행(occurrence)별 most-sub 1개 선택
    picks = []
    for _, r in df_t.iterrows():
        sub = _pick_most_sub_for_row(r.get("pred_top5"), r.get("pred_top5_probs"), t_low)
        if sub is not None and _safe_lower(sub) != t_low:
            picks.append({
                "row_idx": r["row_idx"],
                "year": r["year"],
                "truth": r["truth"],
                "candidate": sub,
            })

    if not picks:
        # 후보가 전혀 안 뽑힌 경우 빈 프레임 반환
        return pd.DataFrame(columns=["truth","year","rank","substitute","weight_sum","weight_norm"])

    pick_df = pd.DataFrame(picks)

    # 2) 포스팅(row_idx, year, truth)별 총 occurrence 수 = n_occ (분모)
    occ_count = (
        df_t.groupby(["row_idx", "year", "truth"])
            .size().rename("n_occ").reset_index()
    )

    # 3) 포스팅 내 후보별 등장횟수(분자): (row_idx, year, truth, candidate) 카운트
    post_counts = (
        pick_df.groupby(["row_idx", "year", "truth", "candidate"])
               .size().rename("cnt").reset_index()
    )

    # 4) 1/n 분배: weight = cnt / n_occ
    post_weights = (
        post_counts.merge(occ_count, on=["row_idx","year","truth"], how="left")
                   .assign(weight=lambda x: x["cnt"] / x["n_occ"].clip(lower=1))
    )

    # 5) 연도별 합산: (year, truth, candidate)별 weight 합
    year_agg = (
        post_weights.groupby(["year", "truth", "candidate"])["weight"]
                    .sum().rename("weight_sum").reset_index()
    )

    # 6) 연도 내 정규화(보기용) & 랭킹(weight_sum 기준)
    results = []
    for year, g in year_agg.groupby("year"):
        denom = g["weight_sum"].sum()
        g = g.assign(weight_norm=(g["weight_sum"] / denom) if denom > 0 else 0.0)
        g = g.sort_values("weight_sum", ascending=False).head(out_topk)
        g = g.assign(rank=range(1, len(g) + 1))
        for _, row in g.iterrows():
            results.append({
                "truth": target_skill,
                "year": year,
                "rank": row["rank"],
                "substitute": row["candidate"],
                "weight_sum": row["weight_sum"],
                "weight_norm": row["weight_norm"],
            })

    if not results:
        return pd.DataFrame(columns=["truth","year","rank","substitute","weight_sum","weight_norm"])

    return pd.DataFrame(results).sort_values(["year", "rank"])

# ------------------------
# CLI
# ------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True, help="Directory containing yearly *with_rowidx.csv files")
    ap.add_argument("--target-skill", required=True, help="Target truth skill to evaluate (case-insensitive)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--out-topk", type=int, default=5, help="Top-K substitutes to keep per year")
    return ap.parse_args()

def main():
    args = parse_args()
    df = load_with_rowidx(args.pred_root)
    print(f"[INFO] Loaded {len(df):,} rows with row_idx files")

    out_df = yearwise_topk_mostsubs_postnorm(
        df,
        target_skill=args.target_skill,
        out_topk=args.out_topk
    )

    out_df.to_csv(args.out, index=False)
    print(f"[OK] Saved → {args.out}")
    print(out_df.head(15))

if __name__ == "__main__":
    main()