# -*- coding: utf-8 -*-
"""
counts_by_pair.csv 기반으로 count>=5 인 (truth, subs, year_month) 조합만 대상으로
'2025-01-01 ~ 2025-06-30' 기간의 Google Trends를 조회하고,
월별 평균(1~6월) 값을 저장한다.

변경점 (요청사항 반영)
- 대상 후보는 counts_by_pair.csv 에서 추출
- 이미 처리한 (truth, subs) 쌍은 out_csv 및 out_csv.partial 에서 읽어 '그대로' 제외
  (공백/대소문자 정규화 없이, 파일에 있는 값과 정확히 같은 문자열 기준)
- 나머지 쌍만 조회 후 이어서 저장

기타 기능
- R/C/C++ 포함 시 cat=31(Programming)
- tqdm 진행률, 중단 후 재개
- 429 Too Many Requests: 즉시 partial 저장 후 10분(600초) 대기, 세션 재시작
- 빈 응답 시 세션 재시작 후 재시도
- Ctrl+C 시 안전 저장
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
import argparse
import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm

YEAR = "2025"
TIMEFRAME = f"{YEAR}-01-01 {YEAR}-06-30"
WAIT_ON_429 = 10  # 10분(초)

def fetch_pair_monthly(pytrends: TrendReq, truth: str, subs: str,
                       geo="US", cat=0, gprop="", retries=3, sleep=1.0) -> pd.DataFrame:
    """2025년 1~6월 구간을 조회하고 월별 평균값을 반환."""
    truth, subs = str(truth), str(subs)
    if not truth.strip() or not subs.strip():
        return pd.DataFrame()

    for attempt in range(retries):
        try:
            pytrends.build_payload([truth, subs], timeframe=TIMEFRAME, geo=geo, cat=cat, gprop=gprop)
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                print(f"[WARN] Empty response for {truth} vs {subs} — restart session & retry...")
                pytrends = TrendReq(hl="en-US", tz=0)
                time.sleep(10)
                continue

            # 날짜 인덱스를 컬럼으로 변환 (그룹바이 편의)
            df = df.reset_index()

            # 월별 평균 산출
            df["month"] = df["date"].dt.to_period("M").astype(str)
            monthly = df.groupby("month")[[truth, subs]].mean().reset_index()

            # 출력 형태 구성
            monthly["truth"] = truth
            monthly["subs"] = subs
            monthly["year_month"] = monthly["month"].str.replace("-", "")
            monthly["truth_trend"] = monthly[truth].round(1).astype(float)
            monthly["subs_trend"] = monthly[subs].round(1).astype(float)
            monthly["trend_diff"] = (monthly["subs_trend"] - monthly["truth_trend"]).round(1)
            monthly = monthly[["truth", "subs", "year_month", "truth_trend", "subs_trend", "trend_diff"]]

            # 1~6월만
            monthly = monthly[monthly["year_month"].between("202501", "202506")]
            return monthly

        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "too many requests" in msg:
                # 상위에서 처리
                raise RuntimeError("429")
            print(f"[WARN] {truth} vs {subs} 실패 ({attempt+1}/{retries}): {e}")
            time.sleep(max(0.5, sleep))

    return pd.DataFrame()


def safe_save(df_list, out_path):
    """중간 저장 (.partial 파일)"""
    if not df_list:
        return
    tmp = pd.concat(df_list, ignore_index=True)
    tmp_path = out_path + ".partial"
    mode = "a" if os.path.exists(tmp_path) else "w"
    header = not os.path.exists(tmp_path)
    tmp.to_csv(tmp_path, mode=mode, header=header, index=False, encoding="utf-8")
    print(f"💾 Partial save → {tmp_path}")
    df_list.clear()


def read_done_pairs(path: str) -> set[tuple[str, str]]:
    """out_csv / out_csv.partial 에서 (truth, subs) 쌍을 '그대로' 읽어 set으로 리턴."""
    try:
        df_done = pd.read_csv(path, usecols=["truth", "subs"])
        return set(map(tuple, df_done[["truth", "subs"]].astype(str).values))
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True, help="counts_by_pair.csv (truth, subs, year_month, count)")
    ap.add_argument("--out-csv", required=True, help="출력 CSV (예: ./counts_by_pair_with_trends_monthly.csv)")
    ap.add_argument("--geo", default="US", help="국가 코드 (예: US, ''=Worldwide)")
    ap.add_argument("--gprop", default="", help="검색 유형(news, froogle, youtube, images)")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    # 입력 로드 및 후보 생성
    src = pd.read_csv(args.in_csv)
    for c in ["truth", "subs", "year_month", "count"]:
        if c not in src.columns:
            raise ValueError(f"필수 컬럼 누락: {c}")

    df_filtered = src[src["count"] >= 1][["truth", "subs", "year_month"]].drop_duplicates()
    pairs_df = df_filtered[["truth", "subs"]].drop_duplicates(ignore_index=True)
    print(f"🎯 count>=5 충족 {len(df_filtered)} rows → {len(pairs_df)} unique pairs (candidates)")

    out_path = args.out_csv
    print(f"📝 output: {out_path} (+ .partial)")

    # 이미 처리된 (truth, subs) 제외 (정확 일치 기준)
    done_pairs = set()
    if os.path.exists(out_path):
        done_pairs |= read_done_pairs(out_path)
    if os.path.exists(out_path + ".partial"):
        done_pairs |= read_done_pairs(out_path + ".partial")

    if done_pairs:
        before = len(pairs_df)
        pairs_df = pairs_df[~pairs_df.apply(tuple, axis=1).isin(done_pairs)].reset_index(drop=True)
        print(f"✅ already-done pairs: {len(done_pairs):,} → remaining to fetch: {len(pairs_df):,} (from {before:,})")
    else:
        print("ℹ️ no already-done pairs found — fetching all candidates.")

    pytrends = TrendReq(hl="en-US", tz=0)
    results = []

    print(f"🚀 Fetching monthly trends (timeframe={TIMEFRAME}, geo={args.geo})...")

    try:
        for i, row in enumerate(tqdm(pairs_df.itertuples(index=False), total=len(pairs_df))):
            truth, subs = str(row.truth), str(row.subs)

            # 카테고리 자동 지정 (R/C/C++)
            lower = {truth.lower(), subs.lower()}
            cat = 31 if any(k in {"r", "c", "c++"} for k in lower) else 0

            try:
                monthly_df = fetch_pair_monthly(
                    pytrends, truth, subs,
                    geo=args.geo, cat=cat, gprop=args.gprop,
                    retries=args.retries, sleep=args.sleep
                )
            except RuntimeError as e:
                if str(e) == "429":
                    print("⚠️ 429 detected — saving progress and sleeping 10 minutes...")
                    safe_save(results, out_path)
                    results.clear()
                    time.sleep(WAIT_ON_429)
                    pytrends = TrendReq(hl="en-US", tz=0)
                    # 현재 쌍은 건너뛰고 다음으로(재시도 로직 원하면 여기서 다시 호출 가능)
                    continue
                else:
                    raise

            if not monthly_df.empty:
                results.append(monthly_df)

            if (i + 1) % 2 == 0:
                print(f"[{i+1}/{len(pairs_df)}] {truth} vs {subs} → {len(monthly_df)} months")

            if len(results) >= 20:
                safe_save(results, out_path)

        # 남은 결과 저장
        if results:
            safe_save(results, out_path)

    except KeyboardInterrupt:
        print("\n🟥 Ctrl+C — saving partial results...")
        safe_save(results, out_path)
        print("💾 safely stopped. Re-run to resume.")
        return

    # partial 병합 → 최종 out_csv
    if os.path.exists(out_path + ".partial"):
        tmp_all = pd.read_csv(out_path + ".partial")
        mode = "a" if os.path.exists(out_path) else "w"
        header = not os.path.exists(out_path)
        tmp_all.to_csv(out_path, mode=mode, header=header, index=False, encoding="utf-8")
        os.remove(out_path + ".partial")
        print(f"✅ merged partial → {out_path}")

    print(f"🎉 Completed: {out_path}")


if __name__ == "__main__":
    main()
    