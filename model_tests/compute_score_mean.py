# -*- coding: utf-8 -*-
"""
LLaMA 기반 SV(sv_llama)의 전체 평균 계산 (FULL VERSION 용)
- bert
- bert_freezed
- w2v (skill2vec)
- cond (conditional prob)

대상 파일 예:
    .../pred/2014/sv_summary_llama_full_bert_2014-07.csv.gz
    .../pred/2020/sv_summary_llama_full_w2v_2020-11.csv.gz
    .../pred/2011/sv_summary_llama_full_conditional_2011-05.csv.gz
"""

import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm

# ---- 모델별 base dir ----
BERT_DIR         = "/home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred"
W2V_DIR          = "/home/jovyan/LEM_data2/hyunjincho/skill2vec_pred_new/pred"
COND_DIR         = "/home/jovyan/LEM_data2/hyunjincho/condprob_pred_new/pred"

CHUNK_SIZE = 200_000
SV_COL     = "sv_llama"


def list_llama_full_files(base_dir: str, model_name: str, years=None):
    """
    base_dir/{year}/sv_summary_llama_full_{model_name}_YYYY-MM.csv(.gz) 파일을 수집
    """
    if not os.path.isdir(base_dir):
        return []

    # 연도 선택
    if years:
        target_years = set(str(y) for y in years)
        year_dirs = sorted([d for d in os.listdir(base_dir)
                            if d.isdigit() and d in target_years])
    else:
        year_dirs = sorted([d for d in os.listdir(base_dir) if d.isdigit()])

    files = []
    for y in year_dirs:
        ypath = os.path.join(base_dir, y)
        if not os.path.isdir(ypath):
            continue

        # FULL 네이밍 파일만 타겟
        pattern_gz  = os.path.join(ypath, f"sv_summary_llama_full_{model_name}_*.csv.gz")
        pattern_csv = os.path.join(ypath, f"sv_summary_llama_full_{model_name}_*.csv")

        f_gz = sorted(glob.glob(pattern_gz))
        f_csv = sorted(glob.glob(pattern_csv))

        files.extend(f_gz + f_csv)

    return files


def compute_llama_global_mean(model_name: str, base_dir: str, years=None):
    files = list_llama_full_files(base_dir, model_name, years)

    if not files:
        print(f"[{model_name}] No full SV files found under: {base_dir}")
        return

    print(f"\n[{model_name}] Found {len(files)} FULL SV files.")

    global_sum = 0.0
    global_cnt = 0

    # 전 모델/연도/파일 전체에서 (truth, masked_sentence)가 중복되지 않도록 전역 set 유지
    seen_keys = set()

    for fp in tqdm(files, desc=f"[{model_name}] scanning", unit="file"):
        if not os.path.isfile(fp):
            continue

        # 필요한 컬럼이 있는지 확인
        try:
            header = pd.read_csv(fp, nrows=0)
        except Exception as e:
            print(f"[{model_name}] Failed reading header: {fp} | {e}")
            continue

        required_cols = {SV_COL, "truth", "masked_sentence"}
        if not required_cols.issubset(header.columns):
            print(f"[{model_name}] Skip (missing one of {required_cols}): {fp}")
            continue

        try:
            # sv_llama, truth, masked_sentence 세 컬럼만 사용
            for chunk in pd.read_csv(fp,
                                     usecols=[SV_COL, "truth", "masked_sentence"],
                                     chunksize=CHUNK_SIZE):

                # sv_llama, truth, masked_sentence 모두 NaN 아닌 것만 사용
                chunk = chunk.dropna(subset=[SV_COL, "truth", "masked_sentence"])
                if chunk.empty:
                    continue

                # 전역 유니크 키 생성: (truth, masked_sentence) 조합
                keys = (
                    chunk["truth"].astype(str)
                    + "|||SEP|||"
                    + chunk["masked_sentence"].astype(str)
                )

                # 이미 본 키는 제거
                is_new = ~keys.isin(seen_keys)
                if not is_new.any():
                    continue

                new_chunk = chunk[is_new]
                new_keys = keys[is_new]

                # 새 키들을 전역 set에 추가
                seen_keys.update(new_keys.tolist())

                # 새롭게 추가된 유니크 (truth, masked_sentence)에 대해서만 sv_llama 합산
                sv = new_chunk[SV_COL]
                if len(sv) == 0:
                    continue

                global_sum += float(sv.sum())
                global_cnt += int(sv.count())

        except Exception as e:
            print(f"[{model_name}] Failed during chunk read: {fp} | {e}")
            continue

    print(f"\n===== [{model_name}] LLaMA SV (FULL, UNIQUE truth+masked_sentence) =====")
    if global_cnt > 0:
        mean_val = global_sum / global_cnt
        print(f"Global unique samples : {global_cnt:,}")
        print(f"Global mean sv_llama  : {mean_val:.4f}")
    else:
        print("No valid sv_llama values found (after dedup).")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", default=None,
                    help="특정 연도만 보고 싶으면 예: --years 2018 2019")

    # ✅ add: dirs override (same flags as score_with_llama.py)
    ap.add_argument("--bert_pred_dir", default=BERT_DIR)
    ap.add_argument("--w2v_pred_dir", default=W2V_DIR)
    ap.add_argument("--cond_pred_dir", default=COND_DIR)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    years = args.years

    compute_llama_global_mean("bert",         args.bert_pred_dir,         years)
    compute_llama_global_mean("w2v",          args.w2v_pred_dir,          years)
    compute_llama_global_mean("conditional",  args.cond_pred_dir,         years)