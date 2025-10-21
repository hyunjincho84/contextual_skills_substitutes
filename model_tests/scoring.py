# -*- coding: utf-8 -*-
"""
SV (Validation Score) 계산 스크립트 — 월별 .csv.gz 단위로 평가/저장
- 입력: pred 디렉토리 아래 연/월별 predictions_YYYY-MM.csv.gz
- 출력: 같은 위치에 sv_summary_YYYY-MM.csv.gz (원본 컬럼 + sv)
"""

import os
import re
import glob
import json
import argparse
from typing import List, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ===== CONFIG =====
DEFAULT_BERT_PRED_DIR = "/home/jovyan/LEM_data2/hyunjincho/bert_pred/pred"
DEFAULT_W2V_PRED_DIR  = "/home/jovyan/LEM_data2/hyunjincho/skill2vec/pred"
DEFAULT_COND_PRED_DIR = "/home/jovyan/LEM_data2/hyunjincho/condprob/pred"

DEFAULT_MLM_CHECKPOINT  = "/home/jovyan/LEM_data2/hyunjincho/bert_pretrained/checkpoint-165687"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN     = 512
WINDOW_SIZE = 256

torch.set_float32_matmul_precision("high")

# -------- 정규화 --------
_PLANG_TAIL = re.compile(r"\s*\(programming language\)\s*$", flags=re.IGNORECASE)
def normalize_skill_text(s: str) -> str:
    s = str(s or "").strip()
    s = _PLANG_TAIL.sub("", s).strip()
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------- (연/월) 파일 목록 수집 --------
def list_monthly_prediction_files(base_dir: str, years: List[str]) -> List[Tuple[str, str]]:
    """
    returns: list of (year_dir_path, file_path) for monthly predictions
    - 허용 패턴: predictions_YYYY-MM.csv.gz (우선), fallback: predictions_YYYY-MM.csv
    - years 지정 시 해당 연도만
    """
    if not os.path.isdir(base_dir):
        return []
    year_dirs = sorted([d for d in os.listdir(base_dir) if d.isdigit()])
    if years:
        want = set(map(str, years))
        year_dirs = [y for y in year_dirs if y in want]

    pairs = []
    for y in year_dirs:
        ypath = os.path.join(base_dir, y)
        if not os.path.isdir(ypath):
            continue
        files = sorted(glob.glob(os.path.join(ypath, "predictions_*[0-9]-[0-9][0-9].csv.gz")))
        if not files:
            files = sorted(glob.glob(os.path.join(ypath, "predictions_*[0-9]-[0-9][0-9].csv")))
        for fp in files:
            pairs.append((ypath, fp))
    return pairs

# -------- 토큰 subseq --------
def _find_subseq(tokens, sub_tokens) -> int:
    if not sub_tokens: return -1
    L, M = len(tokens), len(sub_tokens)
    if M > L: return -1
    for i in range(L - M + 1):
        ok = True
        for j in range(M):
            if tokens[i+j] != sub_tokens[j]:
                ok = False; break
        if ok: return i
    return -1

# -------- 문장 생성 --------
def build_sentence_with_fill(row: pd.Series, fill_skill: str, tokenizer, window_size: int):
    truth_norm = normalize_skill_text(row["truth"])
    fill_skill = normalize_skill_text(fill_skill)

    if "masked_sentence" in row and isinstance(row["masked_sentence"], str):
        sent = str(row["masked_sentence"]).replace("[MASK]", fill_skill)
    elif "sentence" in row and isinstance(row["sentence"], str):
        sent = normalize_skill_text(str(row["sentence"]))
        pat = rf"(?<!\w){re.escape(truth_norm)}(?!\w)"
        sent = re.sub(pat, fill_skill, sent)
    else:
        raise ValueError("원문을 만들 수 없습니다 (masked_sentence 또는 sentence 필요).")

    tokens = tokenizer.tokenize(sent)
    sub_tokens = tokenizer.tokenize(fill_skill)
    idx = _find_subseq(tokens, sub_tokens)

    if idx >= 0:
        start = max(0, idx - window_size)
        end   = min(len(tokens), idx + window_size + len(sub_tokens))
        win_tokens = tokens[start:end]
        fill_start = idx - start
    else:
        win_tokens = tokens
        fill_start = -1

    text = tokenizer.convert_tokens_to_string(win_tokens)
    return text, win_tokens, fill_start

def build_original_sentence(row: pd.Series, tokenizer, window_size: int):
    truth = normalize_skill_text(row["truth"])
    if "masked_sentence" in row and isinstance(row["masked_sentence"], str):
        sent = str(row["masked_sentence"]).replace("[MASK]", truth)
    elif "sentence" in row and isinstance(row["sentence"], str):
        sent = normalize_skill_text(str(row["sentence"]))
    else:
        raise ValueError("원문을 만들 수 없습니다 (masked_sentence 또는 sentence 필요).")

    tokens = tokenizer.tokenize(sent)
    truth_toks = tokenizer.tokenize(truth)
    k = _find_subseq(tokens, truth_toks)

    if k >= 0:
        start = max(0, k - window_size)
        end   = min(len(tokens), k + window_size + len(truth_toks))
        win_tokens = tokens[start:end]
        k_local = k - start
    else:
        win_tokens = tokens
        k_local = -1

    text = tokenizer.convert_tokens_to_string(win_tokens)
    return text, win_tokens, k_local

# -------- SV 계산 --------
@torch.no_grad()
def compute_sv_batch(
    texts_orig, texts_sub, k_indices, tokenizer, model, max_len, sent_batch
):
    assert len(texts_orig) == len(texts_sub) == len(k_indices)
    N = len(texts_orig)
    out = []

    use_amp = (DEVICE == "cuda")

    for s in range(0, N, sent_batch):
        o_batch = texts_orig[s:s+sent_batch]
        p_batch = texts_sub[s:s+sent_batch]
        k_batch = k_indices[s:s+sent_batch]

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            enc_o = tokenizer(o_batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=max_len).to(DEVICE)
        outputs_o = model(**enc_o, output_attentions=True, output_hidden_states=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            enc_p = tokenizer(p_batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=max_len).to(DEVICE)
        outputs_p = model(**enc_p, output_attentions=False, output_hidden_states=True)

        # hidden concat (top-4)
        hs_o = outputs_o.hidden_states
        hs_p = outputs_p.hidden_states
        top4_o = torch.cat(hs_o[-4:], dim=-1)   # (B, T, 4H)
        top4_p = torch.cat(hs_p[-4:], dim=-1)

        # 평균 attention (B, T, T)
        att_stack = torch.stack(outputs_o.attentions, dim=0)  # (L,B,H,T,T)
        mean_att = att_stack.mean(dim=(0,2))                  # (B,T,T)

        attn_mask_o = enc_o["attention_mask"]
        attn_mask_p = enc_p["attention_mask"]

        B = top4_o.size(0)
        for b in range(B):
            k_loc = k_batch[b]
            if k_loc is None or k_loc < 0:
                out.append(float("nan")); continue

            T_o = int(attn_mask_o[b].sum().item())
            T_p = int(attn_mask_p[b].sum().item())
            T   = min(T_o, T_p)

            H_o = top4_o[b, :T, :]
            H_p = top4_p[b, :T, :]

            num = (H_o * H_p).sum(dim=-1)
            den = (H_o.norm(dim=-1) * H_p.norm(dim=-1)).clamp_min(1e-12)
            cos_i = (num / den)  # (T,)

            k_use = min(k_loc, T-1)
            w = mean_att[b, :T, k_use]           # (T,)
            w = torch.nan_to_num(w, 0.0).clamp_min(0.0)
            w = (w / w.sum()) if float(w.sum().item()) > 0 else torch.ones_like(w) / T

            sv_val = float((w * cos_i).sum().item())
            out.append(sv_val)

        # cleanup
        del enc_o, enc_p, outputs_o, outputs_p, hs_o, hs_p, top4_o, top4_p, att_stack, mean_att
        torch.cuda.empty_cache()

    return out

# -------- 한 개 월 파일 평가 --------
def evaluate_one_month_csv(model_name: str, csv_path: str, tokenizer, model, window_size: int, sent_batch: int):
    try:
        df = pd.read_csv(csv_path, engine="python")
    except Exception as e:
        print(f"[{model_name}] FAIL read {csv_path}: {e}")
        return

    needed = {"truth", "pred_top1", "pred_top5"}
    if not needed.issubset(df.columns):
        print(f"[{model_name}] skip {os.path.basename(csv_path)} (missing cols)")
        return

    # 정규화
    for col in ["truth","pred_top1","pred_top5"]:
        if col in df.columns:
            if col == "pred_top5":
                df[col] = df[col].astype(str).map(
                    lambda s: "|".join([normalize_skill_text(x) for x in s.split("|") if str(x).strip()])
                )
            else:
                df[col] = df[col].astype(str).map(normalize_skill_text)

    # 문장 준비
    orig_texts, sub_texts, k_indices = [], [], []
    for _, row in df.iterrows():
        try:
            # 후보 선택 규칙: BERT면 top1==truth일 때 top2, 그 외엔 top1
            truth = normalize_skill_text(row["truth"])
            top1  = normalize_skill_text(row["pred_top1"])
            top5  = [normalize_skill_text(s) for s in str(row["pred_top5"]).split("|") if str(s).strip()]
            chosen = top5[1] if (model_name.lower()=="bert" and top1==truth and len(top5)>=2) else top1

            x_text, _, k_loc = build_original_sentence(row, tokenizer, window_size)
            x_sub , _, _     = build_sentence_with_fill(row, chosen, tokenizer, window_size)

            orig_texts.append(x_text)
            sub_texts.append(x_sub)
            k_indices.append(k_loc)
        except Exception:
            orig_texts.append("")
            sub_texts.append("")
            k_indices.append(-1)

    # SV
    sv_vals = compute_sv_batch(
        texts_orig=orig_texts,
        texts_sub=sub_texts,
        k_indices=k_indices,
        tokenizer=tokenizer,
        model=model,
        max_len=MAX_LEN,
        sent_batch=sent_batch
    )

    out_df = df.copy()
    out_df["sv"] = sv_vals

    # 저장: sv_summary_YYYY-MM.csv.gz
    base = os.path.basename(csv_path)                     # predictions_YYYY-MM.csv[.gz]
    ym = re.findall(r"(\d{4}-\d{2})", base)
    postfix = ym[0] if ym else "month"
    out_name = f"sv_summary_{postfix}.csv.gz"
    out_path = os.path.join(os.path.dirname(csv_path), out_name)
    out_df.to_csv(out_path, index=False, compression="gzip")
    print(f"[{model_name}] saved: {out_path}")

# -------- 메인 --------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-from", type=str, default="all", help="bert,w2v,conditional 또는 all")
    ap.add_argument("--bert_pred_dir", default=DEFAULT_BERT_PRED_DIR)
    ap.add_argument("--w2v_pred_dir",  default=DEFAULT_W2V_PRED_DIR)
    ap.add_argument("--cond_pred_dir", default=DEFAULT_COND_PRED_DIR)
    ap.add_argument("--mlm_ckpt", default=DEFAULT_MLM_CHECKPOINT)
    ap.add_argument("--years", nargs="*", default=[], help="예: 2010 2011 2024 (미지정 시 전체)")
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--sent-batch", type=int, default=32)
    return ap.parse_args()

def main():
    args = parse_args()
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(args.mlm_ckpt)
    model     = AutoModelForMaskedLM.from_pretrained(args.mlm_ckpt).to(DEVICE).eval()

    wanted = set(x.strip().lower() for x in args.start_from.split(",")) if args.start_from != "all" else {"bert","w2v","conditional"}

    model_dirs = []
    if ("bert" in wanted or "all" in wanted) and os.path.isdir(args.bert_pred_dir):
        model_dirs.append(("bert", args.bert_pred_dir))
    if ("w2v" in wanted or "all" in wanted) and os.path.isdir(args.w2v_pred_dir):
        model_dirs.append(("w2v", args.w2v_pred_dir))
    if ("conditional" in wanted or "all" in wanted) and os.path.isdir(args.cond_pred_dir):
        model_dirs.append(("conditional", args.cond_pred_dir))

    if not model_dirs:
        print("예측 디렉토리를 찾을 수 없습니다. --start-from / *_pred_dir 경로를 확인하세요.")
        return

    for name, base_dir in model_dirs:
        month_files = list_monthly_prediction_files(base_dir, args.years)
        if not month_files:
            print(f"[{name}] No monthly predictions under: {base_dir}")
            continue

        for _, fpath in tqdm(month_files, desc=f"[{name}] monthly SV", unit="file"):
            evaluate_one_month_csv(
                model_name=name,
                csv_path=fpath,
                tokenizer=tokenizer,
                model=model,
                window_size=args.window_size,
                sent_batch=args.sent_batch
            )

if __name__ == "__main__":
    main()