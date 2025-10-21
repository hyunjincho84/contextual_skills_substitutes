# -*- coding: utf-8 -*-
"""
.partial 트렌드 결과와 sample_1000_scored.csv를 (truth, subs, year_month) 키로 조인.
- 존재하는 키에 한해 truth_trend, subs_trend, trend_diff를 붙임
- trend_diff - sv_loss_from_norm 값을 계산하여 margin 컬럼으로 저장

기본 경로:
  trends_partial: ./counts_by_pair_with_trends_monthly.csv.partial
  sample_csv    : ./sample_1000_scored.csv
  out_csv       : ./sample_1000_scored_with_trends.csv

사용 예:
  python3 aggregate.py \
    --partial ./counts_by_pair_with_trends_monthly.csv.partial \
    --sample  ./sample_1000_scored.csv \
    --out     ./sample_1000_scored_with_trends.csv
"""

import argparse
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", default="./counts_by_pair_with_trends_monthly.csv.partial",
                    help="부분 저장된 트렌드 파일(.partial) 경로")
    ap.add_argument("--sample", default="./sample_1000_scored.csv",
                    help="샘플 점수 파일 경로 (truth, subs, year_month, sv_loss_from_norm 포함)")
    ap.add_argument("--out", default="./sample_1000_scored_with_trends.csv",
                    help="출력 CSV 경로")
    ap.add_argument("--encoding", default="utf-8", help="입출력 인코딩")
    args = ap.parse_args()

    # 1) 입력 로드
    if not os.path.exists(args.partial):
        # .partial이 없으면 최종본(csv)을 대신 사용하도록 폴백
        alt = args.partial.replace(".partial", "")
        if os.path.exists(alt):
            print(f"⚠️ partial이 없어 최종 파일로 대체: {alt}")
            partial_path = alt
        else:
            raise FileNotFoundError(f"트렌드 파일을 찾을 수 없습니다: {args.partial} (또는 {alt})")
    else:
        partial_path = args.partial

    print(f"📥 Loading trends from: {partial_path}")
    df_tr = pd.read_csv(partial_path, encoding=args.encoding)

    required_tr_cols = {"truth","subs","year_month","truth_trend","subs_trend","trend_diff"}
    if not required_tr_cols.issubset(df_tr.columns):
        raise ValueError(f"트렌드 파일에 필수 컬럼 누락: {required_tr_cols - set(df_tr.columns)}")

    # 중복 키가 있을 수 있으니 키 기준으로 정리 (여러 행이면 평균/마지막 등 정책 선택)
    # 여기서는 동일 키 중 '마지막 행을 채택'하도록 함 (partial append 순서 가정)
    df_tr["__order__"] = range(len(df_tr))
    df_tr = df_tr.sort_values("__order__").drop_duplicates(
        subset=["truth","subs","year_month"], keep="last"
    ).drop(columns="__order__")

    # 2) 샘플 로드
    print(f"📥 Loading sample from: {args.sample}")
    df_samp = pd.read_csv(args.sample, encoding=args.encoding)

    required_s_cols = {"truth","subs","year_month","sv_loss_from_norm"}
    missing_s = required_s_cols - set(df_samp.columns)
    if missing_s:
        raise ValueError(f"샘플 파일에 필수 컬럼 누락: {missing_s}")

    # 키 문자열화(공백/대소문자 이슈 최소화)
    for c in ["truth","subs","year_month"]:
        df_tr[c]   = df_tr[c].astype(str).str.strip()
        df_samp[c] = df_samp[c].astype(str).str.strip()

    # 3) 조인 (sample 기준으로 trends 붙인 후, 매칭 성공한 행만 필터)
    merged = df_samp.merge(
        df_tr[["truth","subs","year_month","truth_trend","subs_trend","trend_diff"]],
        on=["truth","subs","year_month"],
        how="left",
        validate="many_to_one"
    )

    before = len(merged)
    matched = merged["truth_trend"].notna().sum()
    print(f"🔗 조인 결과: 전체 {before:,}행 중 매칭 {matched:,}행")

    # 매칭된 행만 남김
    merged = merged[merged["truth_trend"].notna()].copy()

    # 4) 계산: trend_diff - sv_loss_from_norm
    # 숫자형으로 변환
    for col in ["truth_trend","subs_trend","trend_diff","sv_loss_from_norm"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["margin"] = merged["trend_diff"] - merged["sv_loss_from_norm"]

    # 5) 저장 컬럼 순서(가독성)
    front_cols = ["truth","subs","year_month","truth_trend","subs_trend","trend_diff","sv_loss_from_norm","margin"]
    other_cols = [c for c in merged.columns if c not in front_cols]
    merged = merged[front_cols + other_cols]

    # 6) 저장
    merged.to_csv(args.out, index=False, encoding=args.encoding)
    print(f"✅ Saved: {args.out}")
    print(f"   ➤ 매칭된 row 수: {len(merged):,}")

if __name__ == "__main__":
    main()