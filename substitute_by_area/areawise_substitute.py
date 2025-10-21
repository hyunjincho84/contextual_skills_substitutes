# get_subs_by_mostsubs_postnorm.py
# -*- coding: utf-8 -*-
"""
Target skill에 대해 MajorGroup 단위로 '포스팅 내 truth 중복 보정(post-normalize)'을
확률이 아닌 'most-sub 등장 횟수 비율(=count/n_occ)'로 계산하여
Top-1 (또는 Top-K) 대체 스킬을 추출.

로직 개요
1) row_idx 포함 CSV(*with_rowidx*.csv)들을 읽는다.
2) onet_name → SOC MajorGroup 매핑
3) 타깃 truth에 대해, 각 행(occurrence)에서 'most-sub' 1개만 선택
   - 기본: pred_top5의 Top-1
   - 단, Top-1이 truth면 Top-2 사용
   - 모두 truth이거나 후보 없으면 제외
4) (row_idx, MajorGroup, truth)별 truth 등장 횟수 n_occ 계산
5) (row_idx, MajorGroup, truth, candidate)별 most-sub 등장 횟수 cnt 계산
6) 포스팅내 비율 weight = cnt / n_occ (post-normalize)
7) MajorGroup 단위에서 (truth, candidate)별 weight 합산 → total_weight
8) MajorGroup별 Top-K 후보를 total_weight 기준으로 정렬해 출력
"""

import os, re, glob, math, argparse
import pandas as pd
from tqdm import tqdm

# ------------------------
# Utils
# ------------------------
def normalize_text(s: str) -> str:
    if s is None: return ""
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _find_rowidx_files(pred_root: str):
    files = []
    if os.path.isfile(pred_root):
        files = [pred_root]
    else:
        for year_dir in sorted(glob.glob(os.path.join(pred_root, "[0-9]"*4))):
            files.extend(sorted(glob.glob(os.path.join(year_dir, "*with_rowidx.csv"))))
        if not files:  # 연도폴더 없는 경우
            files = sorted(glob.glob(os.path.join(pred_root, "*with_rowidx.csv")))
    if not files:
        raise FileNotFoundError(f"No *with_rowidx*.csv files under: {pred_root}")
    return files

def _pick_most_sub_for_row(pred_top5: str, truth_lower: str):
    """
    한 행(occurrence)에서 'most-sub' 후보 1개만 선택.
    - 기본: pred_top5의 Top-1
    - 단, Top-1이 truth면 Top-2 사용
    - 둘 다 truth거나 후보가 없으면 None
    """
    cands = [c.strip() for c in str(pred_top5 or "").split("|") if c.strip()]
    if not cands:
        return None
    top1 = cands[0].lower()
    if top1 != truth_lower:
        return cands[0]
    if len(cands) >= 2 and cands[1].lower() != truth_lower:
        return cands[1]
    return None

# ------------------------
# Load + Map MajorGroup
# ------------------------
def load_predictions_with_rowidx(pred_root: str) -> pd.DataFrame:
    usecols = ["row_idx","truth","pred_top5","pred_top5_probs","onet_name","year"]
    files = _find_rowidx_files(pred_root)
    dfs = []
    for fp in tqdm(files, desc="Loading *with_rowidx* predictions"):
        try:
            header = pd.read_csv(fp, nrows=0)
            use = [c for c in usecols if c in header.columns]
            df = pd.read_csv(fp, usecols=use)
            # year 없으면 파일명에서 추출
            if "year" not in df.columns:
                m = re.search(r"(20\d{2})", os.path.basename(fp))
                if m:
                    df["year"] = m.group(1)
            df["__src"] = fp
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] skip {fp}: {e}")
    if not dfs:
        raise FileNotFoundError("No valid CSVs loaded.")
    return pd.concat(dfs, ignore_index=True)

def onet_to_major(df: pd.DataFrame, all_occ_csv: str, soc_major_csv: str) -> pd.DataFrame:
    if "onet_name" not in df.columns:
        raise ValueError("Input must include 'onet_name' for MajorGroup mapping.")

    occ = pd.read_csv(all_occ_csv)
    occ["__key"] = occ["Occupation"].astype(str).map(normalize_text)
    occ_map = dict(zip(occ["__key"], occ["Code"].astype(str)))

    soc = pd.read_csv(soc_major_csv)
    soc["MajorGroup"] = soc["MajorGroup"].astype(str).str.extract(r"(\d{2})").fillna("")
    mg_desc = dict(zip(soc["MajorGroup"], soc["Description"].astype(str)))

    key = df["onet_name"].astype(str).map(normalize_text)
    codes = key.map(lambda k: occ_map.get(k))
    majors = codes.map(lambda c: re.match(r"^(\d{2})-", str(c)).group(1) if isinstance(c,str) and re.match(r"^(\d{2})-", str(c)) else None)
    descs = majors.map(lambda mg: mg_desc.get(mg, None) if mg else None)

    out = df.copy()
    out["MajorGroup"] = majors
    out["MG_Desc"] = descs
    # MajorGroup 매핑 실패는 제외
    return out.dropna(subset=["MajorGroup"])

# ------------------------
# Core (Most-Sub Count-Based Post-Normalization)
# ------------------------
def topk_by_major_mostsubs_postnormalized(
    df: pd.DataFrame,
    target_skill: str,
    out_top_per_mg: int = 1
) -> pd.DataFrame:
    """
    포스팅(row_idx) 내 'most-sub' 등장 횟수 비율(cnt / n_occ)로 post-normalize 후
    MajorGroup 단위로 합산하여 Top-K 산출.
    """
    t_low = normalize_text(target_skill)
    df_t = df[df["truth"].astype(str).str.lower() == t_low]
    if df_t.empty:
        raise ValueError(f"No rows for skill '{target_skill}'")

    # 1) occurrence별 most-sub 1개 선택
    picks = []
    for _, r in df_t.iterrows():
        sub = _pick_most_sub_for_row(r.get("pred_top5"), t_low)
        if sub is not None and normalize_text(sub) != t_low:
            picks.append({
                "row_idx": r["row_idx"],
                "MajorGroup": r["MajorGroup"],
                "MG_Desc": r.get("MG_Desc"),
                "truth": r["truth"],
                "candidate": sub,
                "year": r.get("year"),
            })
    if not picks:
        return pd.DataFrame(columns=["truth","MajorGroup","Description","rank","substitute","total_weight","weight_norm","count_posts","count_norm"])

    pick_df = pd.DataFrame(picks)

    # 2) (row_idx, MajorGroup, truth)별 truth 등장 횟수 = n_occ
    occ_count = (
        df_t.groupby(["row_idx","MajorGroup","truth"], dropna=False)
            .size().rename("n_occ").reset_index()
    )

    # 3) (row_idx, MajorGroup, truth, candidate)별 most-sub 등장 횟수 = cnt
    post_counts = (
        pick_df.groupby(["row_idx","MajorGroup","MG_Desc","truth","candidate"], dropna=False)
               .size().rename("cnt").reset_index()
    )

    # 4) 포스팅내 비율: weight = cnt / n_occ
    post_weights = (
        post_counts.merge(occ_count, on=["row_idx","MajorGroup","truth"], how="left")
                   .assign(weight=lambda x: x["cnt"] / x["n_occ"].clip(lower=1))
    )

    # 5) MajorGroup 단위 누적
    mg_agg = (
        post_weights.groupby(["MajorGroup","MG_Desc","truth","candidate"], dropna=False)["weight"]
                    .agg(total_weight="sum", count_posts="count")
                    .reset_index()
    )

    # 6) 정규화(보기용) & 랭킹
    mg_agg["weight_norm"] = mg_agg["total_weight"] / (
        mg_agg.groupby("MajorGroup")["total_weight"].transform("sum").replace(0, float("nan"))
    )
    mg_agg["count_norm"] = mg_agg["count_posts"] / (
        mg_agg.groupby("MajorGroup")["count_posts"].transform("sum").replace(0, float("nan"))
    )

    mg_agg = mg_agg.sort_values(["MajorGroup","total_weight"], ascending=[True, False])
    mg_agg["rank"] = mg_agg.groupby("MajorGroup")["total_weight"].rank(method="first", ascending=False).astype(int)

    # 7) Top-K per MajorGroup
    mg_topk = (mg_agg[mg_agg["rank"] <= out_top_per_mg]
               .rename(columns={"candidate":"substitute","MG_Desc":"Description"})
               [["truth","MajorGroup","Description","rank","substitute","total_weight","weight_norm","count_posts","count_norm"]]
               .sort_values(["MajorGroup","rank"]))

    return mg_topk

# ------------------------
# CLI
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--target-skill", required=True)
    ap.add_argument("--all-occ", required=True)
    ap.add_argument("--soc-major", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-top-per-mg", type=int, default=1)
    args = ap.parse_args()

    df = load_predictions_with_rowidx(args.pred_root)
    df = onet_to_major(df, args.all_occ, args.soc_major)
    out = topk_by_major_mostsubs_postnormalized(
        df,
        target_skill=args.target_skill,
        out_top_per_mg=args.out_top_per_mg
    )
    out.to_csv(args.out, index=False)
    print(f"[OK] Saved → {args.out}")
    print(out.head(10))

if __name__ == "__main__":
    main()