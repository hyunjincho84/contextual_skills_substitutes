# -*- coding: utf-8 -*-
"""
Query Google Trends for (truth, subs, year_month) pairs with count >= 5
from counts_by_pair.csv, limited to the period
'2025-01-01 ~ 2025-06-30', and store monthly averages (Jan–Jun).
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
WAIT_ON_429 = 600  # 10 minutes (seconds)


def fetch_pair_monthly(pytrends: TrendReq, truth: str, subs: str,
                       geo="US", cat=0, gprop="", retries=3, sleep=1.0) -> pd.DataFrame:
    """Query Jan–Jun 2025 and return monthly average values."""
    truth, subs = str(truth), str(subs)
    if not truth.strip() or not subs.strip():
        return pd.DataFrame()

    for attempt in range(retries):
        try:
            pytrends.build_payload(
                [truth, subs],
                timeframe=TIMEFRAME,
                geo=geo,
                cat=cat,
                gprop=gprop
            )
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                print(f"[WARN] Empty response for {truth} vs {subs} — restarting session & retrying...")
                pytrends = TrendReq(hl="en-US", tz=0)
                time.sleep(10)
                continue

            # Convert date index to column (for groupby)
            df = df.reset_index()

            # Compute monthly averages
            df["month"] = df["date"].dt.to_period("M").astype(str)
            monthly = df.groupby("month")[[truth, subs]].mean().reset_index()

            # Format output
            monthly["truth"] = truth
            monthly["subs"] = subs
            monthly["year_month"] = monthly["month"].str.replace("-", "")
            monthly["truth_trend"] = monthly[truth].round(1).astype(float)
            monthly["subs_trend"] = monthly[subs].round(1).astype(float)
            monthly["trend_diff"] = (monthly["subs_trend"] - monthly["truth_trend"]).round(1)

            monthly = monthly[
                ["truth", "subs", "year_month", "truth_trend", "subs_trend", "trend_diff"]
            ]

            # Keep Jan–Jun only
            monthly = monthly[monthly["year_month"].between("202501", "202506")]
            return monthly

        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "too many requests" in msg:
                # Handled at a higher level
                raise RuntimeError("429")
            print(f"[WARN] {truth} vs {subs} failed ({attempt+1}/{retries}): {e}")
            time.sleep(max(0.5, sleep))

    return pd.DataFrame()


def safe_save(df_list, out_path):
    """Intermediate save to .partial file."""
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
    """Read (truth, subs) pairs exactly as stored in out_csv or out_csv.partial."""
    try:
        df_done = pd.read_csv(path, usecols=["truth", "subs"])
        return set(map(tuple, df_done[["truth", "subs"]].astype(str).values))
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True,
                    help="counts_by_pair.csv (truth, subs, year_month, count)")
    ap.add_argument("--out-csv", required=True,
                    help="Output CSV (e.g., ./counts_by_pair_with_trends_monthly.csv)")
    ap.add_argument("--geo", default="US", help="Geographic code (e.g., US, '' = Worldwide)")
    ap.add_argument("--gprop", default="", help="Search type (news, froogle, youtube, images)")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    # Load input and build candidate pairs
    src = pd.read_csv(args.in_csv)
    for c in ["truth", "subs", "year_month", "count"]:
        if c not in src.columns:
            raise ValueError(f"Missing required column: {c}")

    df_filtered = src[src["count"] >= 1][["truth", "subs", "year_month"]].drop_duplicates()
    pairs_df = df_filtered[["truth", "subs"]].drop_duplicates(ignore_index=True)
    print(
        f"🎯 count>=5 satisfied: {len(df_filtered)} rows "
        f"→ {len(pairs_df)} unique pairs (candidates)"
    )

    out_path = args.out_csv
    print(f"📝 output: {out_path} (+ .partial)")

    # Exclude already-processed pairs (exact match)
    done_pairs = set()
    if os.path.exists(out_path):
        done_pairs |= read_done_pairs(out_path)
    if os.path.exists(out_path + ".partial"):
        done_pairs |= read_done_pairs(out_path + ".partial")

    if done_pairs:
        before = len(pairs_df)
        pairs_df = pairs_df[
            ~pairs_df.apply(tuple, axis=1).isin(done_pairs)
        ].reset_index(drop=True)
        print(
            f"✅ already-done pairs: {len(done_pairs):,} "
            f"→ remaining: {len(pairs_df):,} (from {before:,})"
        )
    else:
        print("ℹ️ No previously processed pairs found — fetching all candidates.")

    pytrends = TrendReq(hl="en-US", tz=0)
    results = []

    print(f"🚀 Fetching monthly trends (timeframe={TIMEFRAME}, geo={args.geo})...")

    try:
        for i, row in enumerate(tqdm(pairs_df.itertuples(index=False), total=len(pairs_df))):
            truth, subs = str(row.truth), str(row.subs)

            # Auto-assign category (Programming) for R/C/C++
            lower = {truth.lower(), subs.lower()}
            cat = 31 if any(k in {"r", "c", "c++"} for k in lower) else 0

            try:
                monthly_df = fetch_pair_monthly(
                    pytrends,
                    truth,
                    subs,
                    geo=args.geo,
                    cat=cat,
                    gprop=args.gprop,
                    retries=args.retries,
                    sleep=args.sleep
                )
            except RuntimeError as e:
                if str(e) == "429":
                    print("⚠️ 429 detected — saving progress and sleeping for 10 minutes...")
                    safe_save(results, out_path)
                    results.clear()
                    time.sleep(WAIT_ON_429)
                    pytrends = TrendReq(hl="en-US", tz=0)
                    continue
                else:
                    raise

            if not monthly_df.empty:
                results.append(monthly_df)

            if (i + 1) % 2 == 0:
                print(f"[{i+1}/{len(pairs_df)}] {truth} vs {subs} → {len(monthly_df)} months")

            if len(results) >= 20:
                safe_save(results, out_path)

        # Save remaining results
        if results:
            safe_save(results, out_path)

    except KeyboardInterrupt:
        print("\n🟥 Ctrl+C detected — saving partial results...")
        safe_save(results, out_path)
        print("💾 Safely stopped. Re-run to resume.")
        return

    # Merge partial into final output
    if os.path.exists(out_path + ".partial"):
        tmp_all = pd.read_csv(out_path + ".partial")
        mode = "a" if os.path.exists(out_path) else "w"
        header = not os.path.exists(out_path)
        tmp_all.to_csv(out_path, mode=mode, header=header, index=False, encoding="utf-8")
        os.remove(out_path + ".partial")
        print(f"✅ Merged partial → {out_path}")

    print(f"🎉 Completed: {out_path}")


if __name__ == "__main__":
    main()