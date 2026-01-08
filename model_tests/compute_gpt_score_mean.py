#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare mean SV between GPT-scored rows vs BERT-scored rows on the SAME sampled indices,
and verify (truth, masked_sentence) equality for matched rows.

GPT inputs (default):
  /home/jovyan/LEM_data2/hyunjincho/gpt_samples/20*/*with_sv_llama.csv.gz
  required cols: year, file, row_idx, sv_llama
  optional (for verification): truth, masked_sentence

BERT source (original format):
  /home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred/{year}/<basename(file)>
  - NO row_idx column (row order is the index)
  - score column could be: sv_llama or sv (auto-detect)
  - for verification, expects truth and masked_sentence columns (common in sv_summary files)

Outputs:
  - GPT: total rows + mean sv_llama
  - BERT: total matched rows + mean sv(_llama)
  - Verification report: mismatch counts (+ a few examples)
"""

import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm


DEFAULT_GPT_PATTERN = "/home/jovyan/LEM_data2/hyunjincho/gpt_samples/20*/*with_sv_llama.csv.gz"
DEFAULT_BERT_BASE   = "/home/jovyan/LEM_data2/hyunjincho/bert_pred_new/pred"


def detect_score_col(cols):
    """Prefer sv_llama if present, else sv."""
    if "sv_llama" in cols:
        return "sv_llama"
    if "sv" in cols:
        return "sv"
    return None


def build_bert_path(bert_base: str, year: str, file_rel: str) -> str:
    # GPT file column example: "2010/sv_summary_llama_full_bert_2010-01.csv.gz"
    base = os.path.basename(str(file_rel))
    return os.path.join(bert_base, str(year), base)


def safe_read_csv_usecols(path: str, cols_wanted):
    """Read only available columns among cols_wanted."""
    hdr = pd.read_csv(path, nrows=0)
    avail = set(hdr.columns)
    use = [c for c in cols_wanted if c in avail]
    if not use:
        return None, avail
    return pd.read_csv(path, usecols=use), avail


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-pattern", type=str, default=DEFAULT_GPT_PATTERN)
    ap.add_argument("--bert-base", type=str, default=DEFAULT_BERT_BASE)
    ap.add_argument("--max-mismatch-examples", type=int, default=5)
    return ap.parse_args()


def main():
    args = parse_args()

    # ----------------------------
    # 1) Load GPT SV outputs
    # ----------------------------
    gpt_files = sorted(glob.glob(args.gpt_pattern))
    if not gpt_files:
        raise FileNotFoundError(f"No GPT files matched: {args.gpt_pattern}")

    print("[STEP 1] Loading GPT SV files...")
    gpt_parts = []
    gpt_has_truth_mask = True

    for fp in tqdm(gpt_files, desc="GPT"):
        # 필수 + (가능하면) truth/masked_sentence까지
        df, avail = safe_read_csv_usecols(
            fp, ["year", "file", "row_idx", "sv_llama", "truth", "masked_sentence"]
        )
        if df is None:
            raise ValueError(f"GPT file missing required cols among year/file/row_idx/sv_llama: {fp}")

        need = {"year", "file", "row_idx", "sv_llama"}
        if not need.issubset(set(df.columns)):
            raise ValueError(f"GPT file missing required {need}: {fp} (have={set(df.columns)})")

        if ("truth" not in df.columns) or ("masked_sentence" not in df.columns):
            gpt_has_truth_mask = False

        df["year"] = df["year"].astype(str)
        df["file"] = df["file"].astype(str)
        df["row_idx"] = df["row_idx"].astype(int)
        gpt_parts.append(df)

    gpt_df = pd.concat(gpt_parts, ignore_index=True)
    gpt_df = gpt_df.dropna(subset=["sv_llama"])

    gpt_cnt = len(gpt_df)
    gpt_mean = float(gpt_df["sv_llama"].mean()) if gpt_cnt > 0 else float("nan")

    print(f"[GPT] Rows    : {gpt_cnt:,}")
    print(f"[GPT] Mean SV : {gpt_mean:.6f}")

    if not gpt_has_truth_mask:
        print("[WARN] GPT files do NOT consistently contain truth/masked_sentence. "
              "Verification will be PARTIAL or SKIPPED for those rows.")

    # group GPT rows by (year, file)
    # also keep mapping row_idx -> (truth, masked_sentence) if available
    gpt_groups = {}
    gpt_verify_maps = {}  # (year,file) -> dict[row_idx] = (truth, masked_sentence)
    for (y, f), sub in gpt_df.groupby(["year", "file"]):
        key = (str(y), str(f))
        gpt_groups[key] = sub["row_idx"].tolist()

        if ("truth" in sub.columns) and ("masked_sentence" in sub.columns):
            m = {}
            # 문자열 normalize
            for _, r in sub.iterrows():
                ri = int(r["row_idx"])
                t = "" if pd.isna(r.get("truth", "")) else str(r.get("truth", ""))
                ms = "" if pd.isna(r.get("masked_sentence", "")) else str(r.get("masked_sentence", ""))
                m[ri] = (t, ms)
            gpt_verify_maps[key] = m

    # ----------------------------
    # 2) Load BERT SV outputs and match by row_idx (row order)
    # ----------------------------
    print("\n[STEP 2] Loading BERT SV files (match by GPT row_idx via row order) + verification...")

    bert_sum = 0.0
    bert_cnt = 0
    used_score_col = None

    miss_files = 0
    miss_rows = 0

    # verification stats
    verify_attempted = 0
    verify_matched = 0
    verify_mismatch = 0
    mismatch_examples = []

    for (year, file_rel), idxs in tqdm(gpt_groups.items(), desc="BERT"):
        bert_path = build_bert_path(args.bert_base, year, file_rel)
        if not os.path.exists(bert_path):
            miss_files += 1
            miss_rows += len(idxs)
            continue

        hdr = pd.read_csv(bert_path, nrows=0)
        cols = set(hdr.columns)

        score_col = detect_score_col(cols)
        if score_col is None:
            miss_files += 1
            miss_rows += len(idxs)
            continue
        if used_score_col is None:
            used_score_col = score_col

        # verification 가능 여부
        can_verify_this_file = ("truth" in cols) and ("masked_sentence" in cols) and ((year, file_rel) in gpt_verify_maps)

        # 필요한 컬럼만 읽기
        usecols = [score_col]
        if can_verify_this_file:
            usecols += ["truth", "masked_sentence"]

        df_bert = pd.read_csv(bert_path, usecols=usecols)
        # score NaN drop은 mean 계산용이지만, row_idx 접근은 원래 row order가 필요해서
        # 여기서는 dropna를 "score만" 기준으로 하되, index가 당겨지면 안됨.
        # 따라서 dropna는 하지 말고, iloc에서 score NaN이면 해당 row는 count에서 빠지게 처리.
        n = len(df_bert)
        if n == 0:
            miss_rows += len(idxs)
            continue

        valid = [i for i in idxs if 0 <= i < n]
        miss_rows += (len(idxs) - len(valid))
        if not valid:
            continue

        # ----- mean 계산 -----
        vals = df_bert[score_col].iloc[valid]
        # NaN 제외
        vals = vals.dropna()
        bert_sum += float(vals.sum())
        bert_cnt += int(vals.count())

        # ----- verification -----
        if can_verify_this_file:
            vmap = gpt_verify_maps[(year, file_rel)]
            # GPT에 있는 row_idx만 검증 시도
            check_idxs = [i for i in valid if i in vmap]
            if check_idxs:
                verify_attempted += len(check_idxs)

                bert_truth = df_bert["truth"].iloc[check_idxs].astype(str).tolist()
                bert_mask  = df_bert["masked_sentence"].iloc[check_idxs].astype(str).tolist()

                for i, bt, bm in zip(check_idxs, bert_truth, bert_mask):
                    gt, gm = vmap[i]
                    if (bt == gt) and (bm == gm):
                        verify_matched += 1
                    else:
                        verify_mismatch += 1
                        if len(mismatch_examples) < args.max_mismatch_examples:
                            mismatch_examples.append({
                                "year": year,
                                "file": file_rel,
                                "row_idx": i,
                                "gpt_truth": gt,
                                "bert_truth": bt,
                                "gpt_masked": gm,
                                "bert_masked": bm,
                            })

    bert_mean = (bert_sum / bert_cnt) if bert_cnt > 0 else float("nan")

    print("\n===== RESULTS =====")
    print(f"[BERT] Score col used : {used_score_col}")
    print(f"[BERT] Matched rows   : {bert_cnt:,}")
    print(f"[BERT] Mean SV        : {bert_mean:.6f}")

    print("\n[Sanity]")
    print(f"Missing BERT files referenced: {miss_files:,}")
    print(f"Missing rows (OOR/empty/etc.) : {miss_rows:,}")

    print("\n===== VERIFICATION (truth, masked_sentence) =====")
    if verify_attempted == 0:
        print("No verification performed (missing truth/masked_sentence in GPT or BERT, or mapping not available).")
    else:
        print(f"Attempted comparisons : {verify_attempted:,}")
        print(f"Matched exactly       : {verify_matched:,}")
        print(f"Mismatched            : {verify_mismatch:,}")
        rate = 100.0 * (verify_matched / verify_attempted)
        print(f"Exact match rate      : {rate:.2f}%")

        if mismatch_examples:
            print("\n--- Mismatch examples (up to N) ---")
            for ex in mismatch_examples:
                print(f"\n[year={ex['year']} row_idx={ex['row_idx']}] file={ex['file']}")
                print(f"  GPT  truth : {ex['gpt_truth']}")
                print(f"  BERT truth : {ex['bert_truth']}")
                print(f"  GPT  masked: {ex['gpt_masked'][:200]}{'...' if len(ex['gpt_masked'])>200 else ''}")
                print(f"  BERT masked: {ex['bert_masked'][:200]}{'...' if len(ex['bert_masked'])>200 else ''}")

    print("\n===== FINAL COMPARISON =====")
    print(f"GPT  : rows={gpt_cnt:,}, mean sv_llama={gpt_mean:.6f}")
    print(f"BERT : rows={bert_cnt:,}, mean {used_score_col}={bert_mean:.6f}")
    if bert_cnt > 0 and gpt_cnt > 0:
        print(f"Δ(GPT - BERT) = {gpt_mean - bert_mean:.6f}")


if __name__ == "__main__":
    main()