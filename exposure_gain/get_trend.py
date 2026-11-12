#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fetch monthly Google Trends proxy values for truth/substitute pairs."""

import argparse
import os
import time
import warnings
from typing import Callable, Optional, Tuple

import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

YEAR = "2025"
TIMEFRAME = f"{YEAR}-01-01 {YEAR}-06-30"
WAIT_ON_429 = 600


class RateLimitError(RuntimeError):
    """Raised when Google Trends returns HTTP 429."""


def create_trend_client(
    connect_timeout: float,
    read_timeout: float,
    session_retries: int,
    session_retry_wait: float,
) -> TrendReq:
    """Create a client while retrying pytrends' initial cookie request."""
    last_error: Optional[Exception] = None
    for attempt in range(1, session_retries + 1):
        try:
            return TrendReq(
                hl="en-US",
                tz=0,
                timeout=(connect_timeout, read_timeout),
            )
        except Exception as exc:
            last_error = exc
            print(
                "[WARN] Google Trends session initialization failed "
                f"({attempt}/{session_retries}): {exc}",
                flush=True,
            )
            if attempt < session_retries:
                wait = session_retry_wait * attempt
                print(
                    f"[WAIT] Retrying session initialization in {wait:g} seconds...",
                    flush=True,
                )
                time.sleep(wait)

    raise RuntimeError(
        "Could not connect to trends.google.com after "
        f"{session_retries} attempts. Verify outbound network access or "
        "increase --connect-timeout/--session-retries."
    ) from last_error


def fetch_pair_monthly(
    pytrends: TrendReq,
    truth: str,
    subs: str,
    client_factory: Callable[[], TrendReq],
    geo: str = "US",
    cat: int = 0,
    gprop: str = "",
    retries: int = 3,
    sleep: float = 1.0,
    timeframe: str = TIMEFRAME,
) -> Tuple[pd.DataFrame, TrendReq]:
    """Query one pair and return its monthly values plus the active client."""
    truth, subs = str(truth).strip(), str(subs).strip()
    if not truth or not subs:
        return pd.DataFrame(), pytrends

    for attempt in range(1, retries + 1):
        try:
            pytrends.build_payload(
                [truth, subs],
                timeframe=timeframe,
                geo=geo,
                cat=cat,
                gprop=gprop,
            )
            frame = pytrends.interest_over_time()
            if frame is None or frame.empty:
                print(
                    f"[WARN] Empty response for {truth} vs {subs} "
                    f"({attempt}/{retries})",
                    flush=True,
                )
                if attempt < retries:
                    time.sleep(max(0.5, sleep))
                    pytrends = client_factory()
                continue

            frame = frame.reset_index()
            frame["month"] = frame["date"].dt.to_period("M").astype(str)
            monthly = frame.groupby("month")[[truth, subs]].mean().reset_index()
            monthly["truth"] = truth
            monthly["subs"] = subs
            monthly["year_month"] = monthly["month"].str.replace("-", "")
            monthly["truth_trend"] = monthly[truth].round(1).astype(float)
            monthly["subs_trend"] = monthly[subs].round(1).astype(float)
            monthly["trend_diff"] = (
                monthly["subs_trend"] - monthly["truth_trend"]
            ).round(1)
            monthly = monthly[
                [
                    "truth",
                    "subs",
                    "year_month",
                    "truth_trend",
                    "subs_trend",
                    "trend_diff",
                ]
            ]

            start_date, end_date = timeframe.split()
            start_month = start_date[:7].replace("-", "")
            end_month = end_date[:7].replace("-", "")
            monthly = monthly[
                monthly["year_month"].between(start_month, end_month)
            ]
            return monthly, pytrends

        except Exception as exc:
            message = str(exc).lower()
            if "429" in message or "too many requests" in message:
                raise RateLimitError("429") from exc

            print(
                f"[WARN] {truth} vs {subs} failed "
                f"({attempt}/{retries}): {exc}",
                flush=True,
            )
            if attempt < retries:
                time.sleep(max(0.5, sleep))
                pytrends = client_factory()

    return pd.DataFrame(), pytrends


def safe_save(frames, out_path: str, encoding: str) -> None:
    """Append buffered results to out_path.partial."""
    if not frames:
        return
    partial_path = out_path + ".partial"
    combined = pd.concat(frames, ignore_index=True)
    exists = os.path.exists(partial_path)
    combined.to_csv(
        partial_path,
        mode="a" if exists else "w",
        header=not exists,
        index=False,
        encoding=encoding,
    )
    print(f"💾 Partial save → {partial_path}", flush=True)
    frames.clear()


def canonical_pair(first: str, second: str) -> tuple[str, str]:
    """Treat (A, B) and (B, A) as the same Google Trends query."""
    first = str(first).strip()
    second = str(second).strip()
    if (first.casefold(), first) <= (second.casefold(), second):
        return first, second
    return second, first


def read_done_pairs(path: str, encoding: str) -> set[tuple[str, str]]:
    """Read completed pairs in canonical order from final/partial output."""
    try:
        done = pd.read_csv(path, usecols=["truth", "subs"], encoding=encoding)
        return {
            canonical_pair(truth, subs)
            for truth, subs in done[["truth", "subs"]].astype(str).values
        }
    except Exception:
        return set()


def merge_partial(out_path: str, encoding: str) -> None:
    """Append the partial file to the final output and remove the partial."""
    partial_path = out_path + ".partial"
    if not os.path.exists(partial_path):
        return
    partial = pd.read_csv(partial_path, encoding=encoding)
    exists = os.path.exists(out_path)
    partial.to_csv(
        out_path,
        mode="a" if exists else "w",
        header=not exists,
        index=False,
        encoding=encoding,
    )
    os.remove(partial_path)
    print(f"✅ Merged partial → {out_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-csv",
        required=True,
        help="Pair counts CSV: truth, subs, year_month, count",
    )
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--geo", default="US")
    parser.add_argument(
        "--gprop", default="", help="Empty, news, froogle, youtube, or images"
    )
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--wait-on-429", type=float, default=WAIT_ON_429)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=3,
                        help="Retries for each truth/substitute pair")
    parser.add_argument("--connect-timeout", type=float, default=15.0,
                        help="Connection timeout in seconds (pytrends default is 2)")
    parser.add_argument("--read-timeout", type=float, default=30.0)
    parser.add_argument("--session-retries", type=int, default=5,
                        help="Retries for initial/replacement TrendReq sessions")
    parser.add_argument("--session-retry-wait", type=float, default=15.0,
                        help="Base wait; retry n waits base*n seconds")
    args = parser.parse_args()

    if args.retries < 1 or args.session_retries < 1:
        parser.error("--retries and --session-retries must be >= 1")
    if args.connect_timeout <= 0 or args.read_timeout <= 0:
        parser.error("timeouts must be > 0")
    if args.sleep < 0 or args.session_retry_wait < 0 or args.wait_on_429 < 0:
        parser.error("wait/sleep values must be >= 0")
    if len(args.timeframe.split()) != 2:
        parser.error("--timeframe must be 'YYYY-MM-DD YYYY-MM-DD'")

    source = pd.read_csv(args.in_csv, encoding=args.encoding)
    required = {"truth", "subs", "year_month", "count"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    canonical = source.apply(
        lambda row: canonical_pair(row["truth"], row["subs"]),
        axis=1,
        result_type="expand",
    )
    source[["truth", "subs"]] = canonical

    filtered = source.loc[
        source["count"] >= args.min_count,
        ["truth", "subs", "year_month"],
    ].drop_duplicates()
    pairs = filtered[["truth", "subs"]].drop_duplicates(ignore_index=True)
    print(
        f"🎯 count>={args.min_count} satisfied: {len(filtered)} rows "
        f"→ {len(pairs)} unique pairs (candidates)"
    )

    out_path = args.out_csv
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"📝 output: {out_path} (+ .partial)")

    done_pairs = set()
    if os.path.exists(out_path):
        done_pairs |= read_done_pairs(out_path, args.encoding)
    if os.path.exists(out_path + ".partial"):
        done_pairs |= read_done_pairs(out_path + ".partial", args.encoding)

    if done_pairs:
        before = len(pairs)
        pairs = pairs[
            ~pairs.apply(tuple, axis=1).isin(done_pairs)
        ].reset_index(drop=True)
        print(
            f"✅ already-done pairs: {len(done_pairs):,} "
            f"→ remaining: {len(pairs):,} (from {before:,})"
        )
    else:
        print("ℹ️ No previously processed pairs found — fetching all candidates.")

    if pairs.empty:
        merge_partial(out_path, args.encoding)
        print(f"🎉 Completed: {out_path}")
        return 0

    def client_factory() -> TrendReq:
        return create_trend_client(
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            session_retries=args.session_retries,
            session_retry_wait=args.session_retry_wait,
        )

    print(
        "🔌 Initializing Google Trends session "
        f"(connect_timeout={args.connect_timeout:g}s, "
        f"read_timeout={args.read_timeout:g}s)...",
        flush=True,
    )
    pytrends = client_factory()
    results = []
    empty_pairs = 0

    print(
        f"🚀 Fetching monthly trends (timeframe={args.timeframe}, geo={args.geo})..."
    )

    try:
        for index, row in enumerate(
            tqdm(pairs.itertuples(index=False), total=len(pairs)), start=1
        ):
            truth, subs = str(row.truth), str(row.subs)
            lower = {truth.lower(), subs.lower()}
            category = 31 if any(k in {"r", "c", "c++"} for k in lower) else 0

            while True:
                try:
                    monthly, pytrends = fetch_pair_monthly(
                        pytrends,
                        truth,
                        subs,
                        client_factory=client_factory,
                        geo=args.geo,
                        cat=category,
                        gprop=args.gprop,
                        retries=args.retries,
                        sleep=args.sleep,
                        timeframe=args.timeframe,
                    )
                    break
                except RateLimitError:
                    print(
                        "⚠️ 429 detected — saving progress and waiting "
                        f"{args.wait_on_429:g} seconds before retrying the same pair...",
                        flush=True,
                    )
                    safe_save(results, out_path, args.encoding)
                    time.sleep(args.wait_on_429)
                    pytrends = client_factory()

            if monthly.empty:
                empty_pairs += 1
            else:
                results.append(monthly)

            if index % 2 == 0:
                print(
                    f"[{index}/{len(pairs)}] {truth} vs {subs} "
                    f"→ {len(monthly)} months"
                )
            if len(results) >= 20:
                safe_save(results, out_path, args.encoding)

        safe_save(results, out_path, args.encoding)

    except KeyboardInterrupt:
        print("\n🟥 Ctrl+C detected — saving partial results...", flush=True)
        safe_save(results, out_path, args.encoding)
        print("💾 Safely stopped. Re-run to resume.", flush=True)
        return 130

    merge_partial(out_path, args.encoding)
    if empty_pairs:
        print(
            f"⚠️ Pairs with no returned Trends rows: {empty_pairs:,}. "
            "They remain eligible for a future retry."
        )
    print(f"🎉 Completed: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
