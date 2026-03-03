#!/usr/bin/env python3
"""
ASX RTO Pre-Announcement Screener
==================================
Detects stocks exhibiting simultaneous abnormal volume AND abnormal price
drift relative to the ASX Small Ords benchmark — the two hallmarks of
informed trading ahead of a reverse-takeover or change-of-activities
announcement.

Signal logic (rolling 90 trading days)
---------------------------------------
  Baseline period : T-90 to T-31  (60 trading days)  — OLS + vol baseline
  Signal window   : T-30 to T-1   (30 trading days)  — live detection zone

  vol_ratio  = mean_volume(signal) / mean_volume(baseline)
  drift      = cumulative AR over signal window
               (actual cum-return minus market-model prediction)

  FLAG raised when:  vol_ratio ≥ VOL_THRESH  AND  drift ≥ DRIFT_THRESH

Usage
-----
  python3 screener.py              # run once, print top-20, save CSV
  python3 screener.py --top 40     # show top 40
  python3 screener.py --vol 1.8 --drift 0.04   # custom thresholds
  python3 screener.py --no-cache   # force re-download (ignore cached prices)

Cron (runs at 6pm AEST Mon-Fri, after ASX close):
  0 8 * * 1-5 /path/to/miniforge3/bin/python3 /path/to/screener.py >> /path/to/screener.log 2>&1
  (8 UTC = 18 AEST / 19 AEDT)
"""

import argparse
import io
import json
import logging
import os
import pathlib
import re
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
SCRIPT_DIR    = pathlib.Path(__file__).parent
CACHE_DIR     = SCRIPT_DIR / ".screener_cache"
OUTPUT_CSV           = SCRIPT_DIR / "screener_output.csv"
DASHBOARD_HTML       = SCRIPT_DIR / "dashboard.html"
WATCHLIST_HISTORY_CSV = SCRIPT_DIR / "watchlist_history.csv"
LAST_RUN_PKL  = SCRIPT_DIR / ".last_run.pkl"   # snapshot for --dashboard-only
LOG_FILE      = SCRIPT_DIR / "screener.log"
BENCHMARK     = "^AXSO"
COMPANIES_URL = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"
SECTORS       = {"Materials", "Energy"}
CALENDAR_DAYS = 200   # fetch window (generous to guarantee 90 trading days)
BATCH_SIZE    = 60    # tickers per yfinance batch request
SLEEP_BATCH   = 1.2   # seconds between batches
VOL_THRESH    = 2.5   # abnormal volume threshold (× baseline)
DRIFT_THRESH  = 0.15  # abnormal drift threshold (15 percentage points)
TOP_N_DEFAULT = 20

# ── Announcement filter ───────────────────────────────────────────────────────
MARKIT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept":     "application/json",
    "Referer":    "https://www.asx.com.au/",
}

# Types that ALWAYS explain an abnormal move (regardless of headline content)
ALWAYS_EXPLAINED_TYPES = {
    "ASSET ACQUISITION & DISPOSAL",  # M&A / asset sale / change in nature
    "ASX QUERY",                     # aware letter — ASX already flagged it publicly
    "Takeover Announcements",        # formal RTO / acquisition announcement
    "Scheme Announcements",          # scheme of arrangement
}

# ISSUED CAPITAL only counts if the headline is a capital-raising action
# (NOT a routine "change of director interest" or "securities on issue" notice)
CAPITAL_KEYWORDS = [
    "placement", "capital raise", "entitlement offer", "entitlement issue",
    "rights issue", "share purchase plan", "trading halt",
]

# PROGRESS REPORT only counts if the headline is an operational field result
# (NOT a strategic advisor appointment, management update, etc.)
DRILLING_KEYWORDS = [
    "drill", "assay", "intercept", "intersect", "metre", "meter",
    "resource", "reserve", "inferred", "indicated", "measured", "jorc",
    "discovery", "mineralisation", "mineralization", "grade", "hole",
]

# M&A / acquisition keywords — trigger EXPLAINED on ANY announcement type
ACQUISITION_KEYWORDS = [
    "acquisition", "merger", "binding", "change in nature",
    "reverse takeover", "change of activities", "proposed transaction",
]

# Trading halt triggers EXPLAINED on ANY announcement type
TRADING_HALT_KEYWORD = "trading halt"

# Signal window ≈ 30 trading days ≈ 45 calendar days
SIGNAL_WINDOW_CALENDAR_DAYS = 45

# ── Market cap filter ─────────────────────────────────────────────────────────
MARKET_CAP_DEFAULT = 50_000_000   # $50M AUD
CAP_CACHE_FILE_TPL = ".screener_cache/market_caps_{date}.pkl"

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Step 1 – Company universe                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_universe() -> pd.DataFrame:
    """Download and filter ASX listed companies to Materials + Energy."""
    log.info("Downloading ASX listed companies …")
    resp = requests.get(COMPANIES_URL,
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=30)
    resp.raise_for_status()

    # Skip the 2-line preamble ("ASX listed companies as at …\n\n")
    text  = resp.text
    lines = text.splitlines()
    # Find the header line (contains "Company name")
    header_idx = next(
        (i for i, l in enumerate(lines) if "Company name" in l), 2
    )
    body = "\n".join(lines[header_idx:])
    df   = pd.read_csv(io.StringIO(body))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Rename to standard names
    col_map = {}
    for c in df.columns:
        if "name" in c:        col_map[c] = "name"
        elif "code" in c:      col_map[c] = "ticker"
        elif "industry" in c or "sector" in c or "gics" in c:
            col_map[c] = "sector"
    df = df.rename(columns=col_map)
    df["ticker_yf"] = df["ticker"].str.strip() + ".AX"
    universe = df[df["sector"].isin(SECTORS)].reset_index(drop=True)
    log.info(f"Universe: {len(universe)} companies  "
             f"({(universe['sector']=='Materials').sum()} Materials, "
             f"{(universe['sector']=='Energy').sum()} Energy)")
    return universe


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Step 2 – Price data download                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def download_prices(tickers: list[str], start: str, end: str,
                    use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """
    Batch-download daily OHLCV for each ticker.
    Returns dict  {ticker_yf: DataFrame(Close, Volume)}
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"prices_{datetime.now(timezone.utc).strftime('%Y%m%d')}.pkl"

    if use_cache and cache_file.exists():
        log.info(f"Loading cached prices from {cache_file.name}")
        return pd.read_pickle(cache_file)

    results: dict[str, pd.DataFrame] = {}
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    log.info(f"Downloading {len(tickers)} tickers in {len(batches)} batches …")

    for b_idx, batch in enumerate(batches):
        log.info(f"  Batch {b_idx+1}/{len(batches)}  ({len(batch)} tickers)")
        try:
            raw = yf.download(
                batch, start=start, end=end,
                auto_adjust=True, progress=False,
                group_by="ticker", threads=True,
            )
        except Exception as e:
            log.warning(f"  Batch {b_idx+1} failed: {e}")
            time.sleep(SLEEP_BATCH * 2)
            continue

        if raw.empty:
            time.sleep(SLEEP_BATCH)
            continue

        # Multi-ticker download → MultiIndex columns  (metric, ticker)
        # Single-ticker download → flat columns
        if isinstance(raw.columns, pd.MultiIndex):
            for tkr in batch:
                try:
                    sub = raw[tkr][["Close", "Volume"]].dropna(how="all")
                    if len(sub) > 10:
                        results[tkr] = sub
                except Exception:
                    pass
        else:
            # Single ticker batch
            tkr = batch[0]
            try:
                sub = raw[["Close", "Volume"]].dropna(how="all")
                if len(sub) > 10:
                    results[tkr] = sub
            except Exception:
                pass

        time.sleep(SLEEP_BATCH)

    log.info(f"Downloaded data for {len(results)}/{len(tickers)} tickers")
    pd.to_pickle(results, cache_file)
    # Clean up old cache files (keep only today's)
    for old in CACHE_DIR.glob("prices_*.pkl"):
        if old != cache_file:
            old.unlink()
    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Step 3 – Signal computation                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_signals(
    prices:   dict[str, pd.DataFrame],
    bm_ret:   pd.Series,
    universe: pd.DataFrame,
    vol_thresh:   float = VOL_THRESH,
    drift_thresh: float = DRIFT_THRESH,
) -> pd.DataFrame:
    """
    For each ticker compute vol_ratio and abnormal_drift.
    Returns DataFrame of all stocks with scores (NaN = insufficient data).
    """
    rows = []

    for _, meta in universe.iterrows():
        tkr = meta["ticker_yf"]
        if tkr not in prices:
            continue

        px  = prices[tkr].copy()
        px.index = pd.to_datetime(px.index).normalize()

        # Align to benchmark (inner join)
        combined = pd.concat(
            [np.log(px["Close"] / px["Close"].shift(1)).rename("ri"),
             bm_ret.rename("rm")],
            axis=1, join="inner"
        ).dropna()

        n = len(combined)
        if n < 75:     # need ≥75 overlapping trading days
            continue

        # Take the most recent 90 trading days
        combined = combined.iloc[-90:]
        n = len(combined)
        if n < 75:
            continue

        baseline = combined.iloc[:60]   # T-90 to T-31
        signal   = combined.iloc[60:]   # T-30 to T-1  (≥15 days needed)
        if len(signal) < 15:
            continue

        # ── Volume ratio ────────────────────────────────────────────────────
        vol_all  = px["Volume"].reindex(combined.index).fillna(0)
        vol_base = vol_all.iloc[:60]
        vol_sig  = vol_all.iloc[60:]
        avg_base = vol_base.replace(0, np.nan).mean()
        avg_sig  = vol_sig.replace(0, np.nan).mean()
        if avg_base < 1 or np.isnan(avg_base) or np.isnan(avg_sig):
            continue
        vol_ratio = avg_sig / avg_base

        # ── Market model OLS on baseline ────────────────────────────────────
        X = np.column_stack([np.ones(len(baseline)), baseline["rm"].values])
        y = baseline["ri"].values
        try:
            betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        alpha, beta = betas

        # ── Abnormal drift over signal window ───────────────────────────────
        ri_sig   = signal["ri"].values
        rm_sig   = signal["rm"].values
        ar_sig   = ri_sig - (alpha + beta * rm_sig)
        cum_ar   = ar_sig.sum()          # log-return cumulative AR
        cum_ret  = ri_sig.sum()          # raw cumulative log-return of stock
        cum_bm   = rm_sig.sum()          # benchmark cumulative log-return

        # R-squared on baseline (model quality)
        y_hat = X @ betas
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        rows.append({
            "ticker":           meta["ticker"],
            "ticker_yf":        tkr,
            "name":             meta.get("name", ""),
            "sector":           meta["sector"],
            "vol_ratio":        round(vol_ratio, 3),
            "abnormal_drift":   round(cum_ar,    4),   # cumulative abnormal log-return
            "cum_ret_pct":      round((np.exp(cum_ret) - 1) * 100, 2),  # actual % return
            "bm_ret_pct":       round((np.exp(cum_bm) - 1)  * 100, 2),
            "beta":             round(beta, 3),
            "r2_baseline":      round(r2,   3),
            "n_baseline":       len(baseline),
            "n_signal":         len(signal),
            "flagged":          (vol_ratio >= vol_thresh) and (cum_ar >= drift_thresh),
            "asx_url":          f"https://www.asx.com.au/markets/company/{meta['ticker']}/announcements",
        })

    df = pd.DataFrame(rows)
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Step 3b – Announcement filter (Markit Digital API)                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _fetch_markit_announcements(asx_code: str) -> list:
    """Fetch up to 5 most-recent announcements from Markit Digital API."""
    url = (f"https://asx.api.markitdigital.com/asx-research/1.0/"
           f"companies/{asx_code}/announcements?count=5&market_sensitive=false")
    try:
        r = requests.get(url, headers=MARKIT_HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        return r.json().get("data", {}).get("items", [])
    except Exception:
        return []


def _parse_ann_date(date_str: str):
    """Parse ISO timestamp from Markit API, strip timezone."""
    try:
        ts = pd.Timestamp(date_str)
        return ts.tz_convert(None) if ts.tzinfo else ts
    except Exception:
        return None


def _classify_ticker(asx_code: str, signal_start: pd.Timestamp) -> dict:
    """
    Fetch and classify announcements for one ticker.

    Returns a dict with:
      explanation_label    : "UNEXPLAINED" | "EXPLAINED" | "UNCLEAR"
      explained_by_type    : announcement type that explains the move (or "")
      explained_by_headline: truncated headline of explaining announcement (or "")
      days_since_material  : days since the most-recent EXPLAINED announcement (or NaN)
      ann_api_note         : short note about API coverage
    """
    items = _fetch_markit_announcements(asx_code)

    if not items:
        return {
            "explanation_label":     "UNCLEAR",
            "explained_by_type":     "",
            "explained_by_headline": "",
            "days_since_material":   float("nan"),
            "ann_api_note":          "no API data",
        }

    today = pd.Timestamp.now()

    # Determine how far back the 5 items reach
    dates = [_parse_ann_date(it.get("date", "")) for it in items]
    valid_dates = [d for d in dates if d is not None]
    oldest = min(valid_dates) if valid_dates else None
    coverage_days = (today - oldest).days if oldest else 0
    full_coverage = coverage_days >= SIGNAL_WINDOW_CALENDAR_DAYS

    # Hard filter: any announcement (of any type) in the last 30 calendar days → skip
    cutoff_30d = today - pd.Timedelta(days=30)
    for d in valid_dates:
        if d >= cutoff_30d:
            return {
                "explanation_label":     "EXPLAINED",
                "explained_by_type":     "RECENT_ANN",
                "explained_by_headline": "Recent announcement (<30d) — hard filtered",
                "days_since_material":   (today - d).days,
                "ann_api_note":          "recent ann (<30d)",
            }

    # Scan items for EXPLAINED announcements within the signal window
    best_type     = ""
    best_headline = ""
    min_days_since = float("nan")

    for it, d in zip(items, dates):
        if d is None or d < signal_start:
            continue  # outside signal window

        ann_type = it.get("announcementType", "")
        headline = it.get("headline", "")
        hl_lower = headline.lower()

        if ann_type in ALWAYS_EXPLAINED_TYPES:
            is_explained = True
        elif ann_type == "ISSUED CAPITAL":
            # Only capital-raising actions — not routine securities notices
            is_explained = any(k in hl_lower for k in CAPITAL_KEYWORDS)
        elif ann_type == "PROGRESS REPORT":
            # Only drilling / resource field results — not advisor appointments
            is_explained = any(k in hl_lower for k in DRILLING_KEYWORDS)
        else:
            # For any other type: only M&A keywords or a trading halt headline
            is_explained = (
                TRADING_HALT_KEYWORD in hl_lower
                or any(k in hl_lower for k in ACQUISITION_KEYWORDS)
            )

        if is_explained:
            days_ago = (today - d).days
            if import_isnan(min_days_since) or days_ago < min_days_since:
                min_days_since = days_ago
                best_type     = ann_type
                best_headline = headline[:80]

    if best_type:
        label = "EXPLAINED"
        note  = f"covered {coverage_days}d"
    elif full_coverage:
        label = "UNEXPLAINED"
        note  = f"covered {coverage_days}d"
    else:
        label = "UNCLEAR"
        note  = f"only {coverage_days}d covered"

    return {
        "explanation_label":     label,
        "explained_by_type":     best_type,
        "explained_by_headline": best_headline,
        "days_since_material":   min_days_since,
        "ann_api_note":          note,
    }


def import_isnan(x) -> bool:
    """Safe isnan that handles non-float values."""
    try:
        return float(x) != float(x)
    except (TypeError, ValueError):
        return False


def annotate_flagged(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each flagged stock, fetch and classify recent announcements.
    Adds five new columns; unflagged rows are left empty/NaN.

    Tier 1 — UNEXPLAINED: flagged AND no material announcement in signal window
             → highest priority for research / regulatory attention
    Tier 2 — UNCLEAR    : flagged AND Markit coverage < 45 days (can't confirm)
    Tier 3 — EXPLAINED  : flagged AND a material public announcement found
             → move is likely legitimate
    """
    signal_start = pd.Timestamp.now() - pd.Timedelta(days=SIGNAL_WINDOW_CALENDAR_DAYS)

    df = df.copy()
    df["explanation_label"]     = ""
    df["explained_by_type"]     = ""
    df["explained_by_headline"] = ""
    df["days_since_material"]   = float("nan")
    df["ann_api_note"]          = ""

    flagged_idx = df.index[df["flagged"]].tolist()
    n = len(flagged_idx)
    log.info(f"Fetching announcements for {n} flagged stocks …")

    for i, idx in enumerate(flagged_idx):
        asx_code = df.at[idx, "ticker"]
        result   = _classify_ticker(asx_code, signal_start)
        for col, val in result.items():
            df.at[idx, col] = val
        if (i + 1) % 10 == 0 or (i + 1) == n:
            log.info(f"  Announcements: {i+1}/{n} done")
        time.sleep(0.2)   # polite rate limit

    counts = df.loc[df["flagged"], "explanation_label"].value_counts().to_dict()
    log.info(f"Annotation complete — " +
             " | ".join(f"{v} {k}" for k, v in counts.items()))
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Step 4 – Ranking & output                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

RANK_WEIGHTS = {"vol_ratio": 0.4, "abnormal_drift": 0.6}

# Tier order for sort (lower = higher priority)
_TIER_ORDER = {"UNEXPLAINED": 0, "UNCLEAR": 1, "EXPLAINED": 2, "": 3}


def rank_results(df: pd.DataFrame, use_ann_filter: bool = False) -> pd.DataFrame:
    """
    Composite score for ranking:
      score = 0.4 × z(vol_ratio) + 0.6 × z(abnormal_drift)

    When use_ann_filter=True the sort is:
      primary  — tier  (UNEXPLAINED → UNCLEAR → EXPLAINED → unflagged)
      secondary — score (descending within tier)
    """
    if df.empty:
        return df
    for col in ["vol_ratio", "abnormal_drift"]:
        mu  = df[col].mean()
        sd  = df[col].std()
        df[f"z_{col}"] = (df[col] - mu) / (sd + 1e-9)
    df["score"] = (RANK_WEIGHTS["vol_ratio"]      * df["z_vol_ratio"] +
                   RANK_WEIGHTS["abnormal_drift"]  * df["z_abnormal_drift"])

    if use_ann_filter and "explanation_label" in df.columns:
        def _tier(r):
            if not r["flagged"]:
                return 4
            return _TIER_ORDER.get(r.get("explanation_label", ""), 3)
        df["tier"] = df.apply(_tier, axis=1)
        return df.sort_values(["tier", "score"],
                              ascending=[True, False]).reset_index(drop=True)

    return df.sort_values("score", ascending=False).reset_index(drop=True)


_TIER_LABELS = {
    0: ("UNEXPLAINED SIGNAL", "\033[91m"),   # red   — Tier 1, investigate
    1: ("UNCLEAR (limited API)","\033[93m"), # yellow — Tier 2, limited data
    2: ("EXPLAINED",           "\033[92m"),  # green  — Tier 3, legit move
    3: ("(unflagged)",         "\033[0m"),   # normal
    4: ("(unflagged)",         "\033[0m"),
}
_RESET = "\033[0m"


def print_results(df: pd.DataFrame, top_n: int, ts: str,
                  vol_thresh: float, drift_thresh: float,
                  use_ann_filter: bool = False) -> None:
    flagged   = df[df["flagged"]].head(top_n)
    unflagged = df[~df["flagged"]].head(max(0, top_n - len(flagged)))
    display   = pd.concat([flagged, unflagged]).head(top_n)

    bar  = "═" * 120
    bar2 = "─" * 120
    print(f"\n{bar}")
    print(f"  ASX RTO SCREENER  —  {ts}")
    print(f"  Thresholds: vol_ratio ≥ {vol_thresh}×   |   abnormal drift ≥ {drift_thresh*100:.1f}%")
    n_flagged = int(df["flagged"].sum())
    if use_ann_filter and "explanation_label" in df.columns:
        counts = df.loc[df["flagged"], "explanation_label"].value_counts().to_dict()
        tier_summary = "  ".join(f"{counts.get(k,0)} {k}"
                                 for k in ("UNEXPLAINED","UNCLEAR","EXPLAINED"))
        print(f"  Universe  : {len(df)} screened  |  Flagged: {n_flagged}  →  {tier_summary}")
        print(f"  Tier 1 (UNEXPLAINED) = no material public announcement found in last ~30 trading days")
    else:
        print(f"  Universe  : {len(df)} companies screened   |   Flagged : {n_flagged}")
    print(f"{bar}")

    header = (
        f"{'Rank':>4}  {'Ticker':<6}  {'Score':>6}  "
        f"{'VolRatio':>9}  {'Drift%':>8}  {'Ret%':>7}  "
        f"{'Beta':>5}  {'R²':>5}  {'Sector':<12}  {'Flag':>5}  "
        f"{'Label':<14}  Company"
    )
    print(header)
    print(bar2)

    prev_tier = None
    for rank, (_, r) in enumerate(display.iterrows(), 1):
        flag_str  = "🚩 YES" if r["flagged"] else "  no"
        drift_pct = round(r["abnormal_drift"] * 100, 2)
        tier      = int(r.get("tier", 4 if not r["flagged"] else 3))

        # Print tier separator when tier changes (only in ann-filter mode)
        if use_ann_filter and tier != prev_tier:
            tier_label, tier_color = _TIER_LABELS.get(tier, ("", "\033[0m"))
            if r["flagged"]:
                print(f"\n  {tier_color}── {tier_label} ──{_RESET}")
            prev_tier = tier

        ann_label = r.get("explanation_label", "") if use_ann_filter else ""
        _, color  = _TIER_LABELS.get(tier, ("", "\033[0m"))
        label_str = f"{color}{ann_label:<14}{_RESET}" if ann_label else f"{'':14}"

        print(
            f"{rank:>4}  {r['ticker']:<6}  {r['score']:>+6.3f}  "
            f"{r['vol_ratio']:>9.2f}×  {drift_pct:>+8.2f}%  "
            f"{r['cum_ret_pct']:>+7.2f}%  "
            f"{r['beta']:>5.2f}  {r['r2_baseline']:>5.3f}  "
            f"{r['sector']:<12}  {flag_str:<5}  "
            f"{label_str}  {str(r['name'])[:26]:<26}"
        )
        # Second line: URL + explanation note if available
        note = ""
        if use_ann_filter and r.get("explained_by_headline"):
            note = f"  ← {r['explained_by_type']}: {r['explained_by_headline'][:60]}"
        elif use_ann_filter and r.get("ann_api_note"):
            note = f"  [{r['ann_api_note']}]"
        indent = " " * 8
        print(f"{indent}\033[94m{r['asx_url']}\033[0m{note}")
        print()

    print(bar)
    print(f"  Full results → {OUTPUT_CSV}")
    print(bar + "\n")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="ASX RTO Pre-Announcement Screener")
    parser.add_argument("--top",           type=int,   default=TOP_N_DEFAULT, help="Number of top results to print")
    parser.add_argument("--vol",           type=float, default=VOL_THRESH,    help="Volume ratio threshold (default 2.0)")
    parser.add_argument("--drift",         type=float, default=DRIFT_THRESH,  help="Abnormal drift threshold (default 0.05 = 5%%)")
    parser.add_argument("--no-cache",      action="store_true",               help="Force re-download, ignore cache")
    parser.add_argument("--sectors",       nargs="+",  default=list(SECTORS), help="Sectors to include (default Materials Energy)")
    parser.add_argument("--output",        type=str,   default=str(OUTPUT_CSV),help="Output CSV path")
    parser.add_argument("--no-ann-filter",  action="store_true",   help="Skip announcement filter (faster, no Markit API calls)")
    parser.add_argument("--dashboard-only", action="store_true",   help="Regenerate dashboard.html from last saved run; skip screener")
    parser.add_argument("--max-cap",        type=float, default=MARKET_CAP_DEFAULT,
                        help="Max market cap AUD (default 50000000 = $50M). 0 = no filter.")
    args = parser.parse_args()

    use_ann_filter = not args.no_ann_filter

    if args.dashboard_only:
        log.info("--dashboard-only: dashboard.html is now static; open it in your browser.")
        return

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log.info(f"{'='*60}")
    log.info(f"ASX RTO Screener run at {ts}")
    log.info(f"Thresholds: vol≥{args.vol}×  drift≥{args.drift*100:.1f}%  top={args.top}")
    log.info(f"Ann-filter: {'ON' if use_ann_filter else 'OFF (--no-ann-filter)'}")
    log.info(f"Market cap: {'≤ $' + str(int(args.max_cap/1e6)) + 'M' if args.max_cap > 0 else 'no filter'}")
    log.info(f"{'='*60}")

    # ── 1. Universe ───────────────────────────────────────────────────────
    universe = load_universe()
    universe = universe[universe["sector"].isin(args.sectors)].reset_index(drop=True)

    # ── 2. Date range ─────────────────────────────────────────────────────
    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=CALENDAR_DAYS)
    start    = start_dt.strftime("%Y-%m-%d")
    end      = end_dt.strftime("%Y-%m-%d")
    log.info(f"Price window: {start} → {end}  ({CALENDAR_DAYS} calendar days)")

    # ── 3. Benchmark ──────────────────────────────────────────────────────
    log.info(f"Downloading benchmark {BENCHMARK} …")
    bm_raw = yf.download(BENCHMARK, start=start, end=end,
                          auto_adjust=True, progress=False)
    if bm_raw.empty:
        log.error("Could not fetch benchmark data — aborting.")
        sys.exit(1)
    if isinstance(bm_raw.columns, pd.MultiIndex):
        bm_raw.columns = bm_raw.columns.get_level_values(0)
    bm_ret = np.log(bm_raw["Close"] / bm_raw["Close"].shift(1)).dropna()
    bm_ret.index = pd.to_datetime(bm_ret.index).normalize()
    log.info(f"Benchmark: {len(bm_ret)} trading days")

    # ── 4. Stock prices ───────────────────────────────────────────────────
    tickers  = universe["ticker_yf"].tolist()
    prices   = download_prices(tickers, start, end,
                                use_cache=not args.no_cache)

    # ── 5. Compute signals ────────────────────────────────────────────────
    log.info("Computing signals …")
    t_signal_start = time.time()
    results = compute_signals(prices, bm_ret, universe,
                               vol_thresh=args.vol, drift_thresh=args.drift)
    log.info(f"Signals computed in {time.time()-t_signal_start:.1f}s  "
             f"({len(results)} stocks scored, {results['flagged'].sum()} flagged)")

    if results.empty:
        log.warning("No results — check data availability.")
        return

    # ── 6. Market cap filter ──────────────────────────────────────────────
    if args.max_cap > 0:
        results = apply_market_cap_filter(results, args.max_cap,
                                          use_cache=not args.no_cache)
    else:
        results["market_cap_aud"] = float("nan")

    # ── 6b. Announcement filter ───────────────────────────────────────────
    if use_ann_filter:
        results = annotate_flagged(results)

    # ── 7. Rank ───────────────────────────────────────────────────────────
    results = rank_results(results, use_ann_filter=use_ann_filter)

    # ── 8. Print ──────────────────────────────────────────────────────────
    print_results(results, top_n=args.top, ts=ts,
                  vol_thresh=args.vol, drift_thresh=args.drift,
                  use_ann_filter=use_ann_filter)

    # ── 9. Save CSV ───────────────────────────────────────────────────────
    out_path  = pathlib.Path(args.output)
    results["run_timestamp"]   = ts
    results["vol_threshold"]   = args.vol
    results["drift_threshold"] = args.drift
    results["ann_filter_on"]   = use_ann_filter

    # Archive CSV if schema changed (market_cap_aud column added)
    _migrate_csv_if_needed(out_path)

    # Append to CSV (for historical tracking)
    write_header = not out_path.exists()
    results.to_csv(out_path, mode="a", header=write_header, index=False)
    log.info(f"Results appended to {out_path}")

    # ── 10. Summary stats ─────────────────────────────────────────────────
    flagged_df = results[results["flagged"]]
    top_ticker = flagged_df.iloc[0]["ticker"] if len(flagged_df) else "none"
    if use_ann_filter and "explanation_label" in flagged_df.columns:
        n_unexplained = (flagged_df["explanation_label"] == "UNEXPLAINED").sum()
        n_unclear     = (flagged_df["explanation_label"] == "UNCLEAR").sum()
        n_explained   = (flagged_df["explanation_label"] == "EXPLAINED").sum()
        log.info(f"Run complete. Flagged: {len(flagged_df)} "
                 f"(Unexplained={n_unexplained}, Unclear={n_unclear}, Explained={n_explained}) "
                 f"| Top Tier-1: {top_ticker}")
    else:
        log.info(f"Run complete. Flagged: {len(flagged_df)} | Top ticker: {top_ticker}")

    # ── 11. Save snapshot + bake dashboard ───────────────────────────────
    pd.to_pickle({"df": results, "ts": ts}, LAST_RUN_PKL)
    update_watchlist_history(results, ts)
    bake_dashboard(results, ts)

    # Print cron setup hint on first run
    if not (SCRIPT_DIR / ".cron_hint_shown").exists():
        _print_cron_hint()
        (SCRIPT_DIR / ".cron_hint_shown").touch()


def _print_cron_hint():
    python  = sys.executable
    script  = pathlib.Path(__file__).resolve()
    logfile = LOG_FILE.resolve()
    print("\n" + "─" * 70)
    print("CRON SETUP  (runs daily at 6pm AEST = 8 UTC, Mon–Fri)")
    print("─" * 70)
    print("Add this line to your crontab  (run: crontab -e)")
    print()
    print(f"  0 8 * * 1-5 {python} {script} >> {logfile} 2>&1")
    print()
    print("Or for AEST timezone-aware scheduling via launchd (macOS):")
    print(f"  Use 'Automator' or 'cron' with TZ=Australia/Sydney")
    print("─" * 70 + "\n")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Step 3c – Market cap filter                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _fetch_one_market_cap(ticker_yf: str) -> float | None:
    """Fetch market cap (AUD) for a single ticker via yfinance."""
    try:
        t   = yf.Ticker(ticker_yf)
        cap = getattr(t.fast_info, "market_cap", None)
        if not cap:
            cap = t.info.get("marketCap") or t.info.get("market_cap")
        return float(cap) if cap else None
    except Exception:
        return None


def fetch_market_caps(tickers: list[str], use_cache: bool = True) -> dict[str, float | None]:
    """
    Fetch market caps for a list of tickers.
    Caches results per calendar day in .screener_cache/.
    Returns {ticker_yf: cap_aud_or_None}.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    today      = datetime.now(timezone.utc).strftime("%Y%m%d")
    cache_file = CACHE_DIR / f"market_caps_{today}.pkl"

    cached: dict = {}
    if use_cache and cache_file.exists():
        cached = pd.read_pickle(cache_file)

    missing = [t for t in tickers if t not in cached]
    if missing:
        log.info(f"Fetching market caps for {len(missing)} tickers …")
        for i, tkr in enumerate(missing):
            cached[tkr] = _fetch_one_market_cap(tkr)
            if (i + 1) % 10 == 0:
                log.info(f"  Market caps: {i+1}/{len(missing)} done")
            time.sleep(0.35)
        pd.to_pickle(cached, cache_file)
        # Clean stale cache files
        for old in CACHE_DIR.glob("market_caps_*.pkl"):
            if old != cache_file:
                old.unlink()

    return {t: cached.get(t) for t in tickers}


def apply_market_cap_filter(df: pd.DataFrame, max_cap: float,
                             use_cache: bool = True) -> pd.DataFrame:
    """
    For every flagged stock: fetch market cap and add market_cap_aud column.
    Stocks with missing cap OR cap > max_cap are de-flagged (excluded from output).
    Unflagged stocks get NaN market_cap_aud (no API call wasted).
    """
    df = df.copy()
    df["market_cap_aud"] = float("nan")

    flagged_idx  = df.index[df["flagged"]].tolist()
    flagged_tkrs = [df.at[i, "ticker_yf"] for i in flagged_idx]

    caps = fetch_market_caps(flagged_tkrs, use_cache=use_cache)

    n_before = len(flagged_idx)
    for idx, tkr in zip(flagged_idx, flagged_tkrs):
        cap = caps.get(tkr)
        df.at[idx, "market_cap_aud"] = float(cap) if cap else float("nan")
        if cap is None or cap > max_cap:
            df.at[idx, "flagged"] = False   # de-flag: above limit or unknown

    n_after   = int(df["flagged"].sum())
    n_removed = n_before - n_after
    log.info(f"Market cap filter (≤${max_cap/1e6:.0f}M): "
             f"{n_removed} removed, {n_after} flagged remain")
    return df


def update_watchlist_history(results: pd.DataFrame, ts: str) -> None:
    """
    Append tickers to watchlist_history.csv only if they have appeared as
    UNEXPLAINED for 3 consecutive daily runs (including today's).
    """
    if not OUTPUT_CSV.exists():
        return

    try:
        history = pd.read_csv(OUTPUT_CSV)
    except Exception:
        return

    if "run_timestamp" not in history.columns or "explanation_label" not in history.columns:
        return

    # Get the 3 most-recent distinct run timestamps (today's run is already appended)
    all_ts = sorted(history["run_timestamp"].dropna().unique())
    if len(all_ts) < 3:
        return  # not enough history yet
    last_3 = all_ts[-3:]

    # For each of the 3 runs, collect UNEXPLAINED tickers
    sets = []
    for run_ts in last_3:
        mask = (history["run_timestamp"] == run_ts) & (history["explanation_label"] == "UNEXPLAINED")
        sets.append(set(history.loc[mask, "ticker"].tolist()))

    # Only tickers UNEXPLAINED in all 3 consecutive runs
    consecutive = sets[0] & sets[1] & sets[2]
    if not consecutive:
        return

    flagged = results[results["flagged"] & results["ticker"].isin(consecutive)].copy()
    if flagged.empty:
        return

    watch_cols = ["ticker", "name", "explanation_label", "vol_ratio",
                  "abnormal_drift", "days_since_material", "market_cap_aud",
                  "asx_url", "run_timestamp"]
    for col in watch_cols:
        if col not in flagged.columns:
            flagged[col] = ""
    flagged["run_timestamp"] = ts

    write_header = not WATCHLIST_HISTORY_CSV.exists()
    flagged[watch_cols].to_csv(WATCHLIST_HISTORY_CSV, mode="a",
                               header=write_header, index=False)
    log.info(f"Watchlist updated: {sorted(consecutive)} → {WATCHLIST_HISTORY_CSV.name}")


# Columns baked into dashboard.html (slim subset for fast page load)
_DASH_COLS = ["ticker", "name", "explanation_label", "vol_ratio",
              "abnormal_drift", "days_since_material", "market_cap_aud",
              "asx_url", "run_timestamp"]


def bake_dashboard(results: pd.DataFrame, ts: str) -> None:
    """
    Embed the latest run's data directly into dashboard.html so it opens
    instantly on any device — no server, no drag-and-drop required.
    Replaces the /* @@BAKED_CSV_START@@ */ … /* @@BAKED_CSV_END@@ */ block.
    """
    if not DASHBOARD_HTML.exists():
        log.warning("dashboard.html not found — skipping bake")
        return

    flagged = results[results["flagged"]].copy()
    out_cols = [c for c in _DASH_COLS if c in flagged.columns]
    csv_str = flagged[out_cols].to_csv(index=False)

    replacement = (
        "/* @@BAKED_CSV_START@@ */\n"
        f"var BAKED_CSV = {json.dumps(csv_str)};\n"
        "/* @@BAKED_CSV_END@@ */"
    )

    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    html = re.sub(
        r"/\* @@BAKED_CSV_START@@ \*/.*?/\* @@BAKED_CSV_END@@ \*/",
        replacement,
        html,
        flags=re.DOTALL,
    )
    DASHBOARD_HTML.write_text(html, encoding="utf-8")
    log.info(f"Dashboard baked: {len(flagged)} flagged stocks → {DASHBOARD_HTML.name}")


def _migrate_csv_if_needed(out_path: pathlib.Path) -> None:
    """
    If screener_output.csv was created before market_cap_aud was added, archive
    it and start fresh so the dashboard gets a consistent column schema.
    """
    if not out_path.exists():
        return
    try:
        cols = set(pd.read_csv(out_path, nrows=0).columns)
        if "market_cap_aud" not in cols:
            archive = out_path.with_name("screener_output.archive.csv")
            out_path.rename(archive)
            log.info(f"Archived old-schema CSV → {archive.name}  (market_cap_aud added)")
    except Exception as e:
        log.warning(f"CSV schema check failed: {e}")

# Dashboard: open dashboard.html in your browser (reads screener_output.csv via FileReader)

_OBSOLETE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASX RTO Screener</title>
<style>
:root {
  --bg:     #0d1117; --surf:   #161b22; --surf2: #1c2128;
  --brd:    #30363d; --brd2:   #21262d;
  --txt:    #e6edf3; --mut:    #8b949e;
  --red:    #f85149; --redbg:  rgba(248,81,73,.10);
  --amb:    #e3b341; --ambbg:  rgba(227,179,65,.10);
  --grn:    #3fb950; --grnbg:  rgba(63,185,80,.10);
  --blu:    #58a6ff;
  --mono:   'SF Mono','Fira Code',Consolas,monospace;
  --sans:   -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);
  font-size:14px;line-height:1.6;padding:28px 36px;max-width:1500px;margin:0 auto}

/* Header */
.hdr{display:flex;justify-content:space-between;align-items:flex-start;
  padding-bottom:20px;margin-bottom:24px;border-bottom:1px solid var(--brd)}
.hdr h1{font-size:20px;font-weight:700;letter-spacing:-.3px}
.hdr .sub{color:var(--mut);font-size:13px;margin-top:3px}
.hdr-right{text-align:right;font-size:13px;color:var(--mut)}
.hdr-right strong{color:var(--txt);font-weight:600}

/* Stats chips */
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:28px;align-items:center}
.chip{display:inline-flex;align-items:center;gap:6px;padding:5px 14px;
  border-radius:20px;font-size:13px;font-weight:500;border:1px solid var(--brd);
  background:var(--surf);color:var(--mut)}
.chip .n{font-family:var(--mono);font-size:15px;font-weight:700}
.chip-r{border-color:var(--red);background:var(--redbg);color:var(--red)}
.chip-a{border-color:var(--amb);background:var(--ambbg);color:var(--amb)}
.chip-g{border-color:var(--grn);background:var(--grnbg);color:var(--grn)}

/* Section */
.section{margin-bottom:24px}
.sec-hdr{display:flex;align-items:center;gap:10px;padding:10px 0;
  cursor:pointer;user-select:none;border-bottom:1px solid var(--brd)}
.sec-hdr h2{font-size:11px;font-weight:700;letter-spacing:.7px;text-transform:uppercase}
.sec-hdr .badge{font-family:var(--mono);font-size:12px;font-weight:700;
  padding:2px 9px;border-radius:10px}
.sec-hdr .note{font-size:12px;color:var(--mut);flex:1}
.sec-hdr .chev{color:var(--mut);font-size:13px;margin-left:auto;transition:transform .2s}
.sec-hdr.shut .chev{transform:rotate(-90deg)}
.t1 h2{color:var(--red)} .t1 .badge{background:var(--redbg);color:var(--red)}
.t2 h2{color:var(--amb)} .t2 .badge{background:var(--ambbg);color:var(--amb)}
.t3 h2{color:var(--grn)} .t3 .badge{background:var(--grnbg);color:var(--grn)}
.sec-body{overflow:hidden}

/* Table */
.tbl-wrap{border:1px solid var(--brd);border-top:none;border-radius:0 0 8px 8px;overflow-x:auto}
table{width:100%;border-collapse:collapse;min-width:680px}
thead tr{background:var(--surf)}
thead th{padding:9px 14px;font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:.5px;color:var(--mut);text-align:left;
  border-bottom:1px solid var(--brd);cursor:pointer;white-space:nowrap;user-select:none}
thead th:hover{color:var(--txt)}
thead th.r{text-align:right}
thead th.asc::after{content:' ↑';opacity:.7}
thead th.desc::after{content:' ↓';opacity:.7}
tbody tr{border-bottom:1px solid var(--brd2)}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:rgba(255,255,255,.03)}
td{padding:10px 14px;vertical-align:middle}
td.r{text-align:right;font-family:var(--mono);font-size:13px}
.tkr a{color:var(--blu);text-decoration:none;font-weight:700;font-size:14px;font-family:var(--mono)}
.tkr a:hover{text-decoration:underline}
.pos{color:var(--grn)} .neg{color:var(--red)} .dim{color:var(--mut)}
.bold{font-weight:600}
.nm{max-width:190px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cov{display:inline-block;font-size:11px;padding:2px 8px;border-radius:10px;
  background:rgba(139,148,158,.12);color:var(--mut);font-family:var(--mono);white-space:nowrap}
.extyp{display:inline-block;font-size:10px;font-weight:700;padding:1px 6px;
  border-radius:4px;background:var(--grnbg);color:var(--grn);margin-right:5px;
  text-transform:uppercase;white-space:nowrap}
.exhl{font-size:12px;color:var(--mut);max-width:280px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;display:inline-block;vertical-align:middle}
.empty{padding:28px;text-align:center;color:var(--mut);font-size:13px}

/* Footer */
.footer{border-top:1px solid var(--brd);padding-top:20px;margin-top:8px;
  display:grid;grid-template-columns:1fr 1fr;gap:28px}
.footer h3{font-size:11px;color:var(--mut);text-transform:uppercase;
  letter-spacing:.5px;margin-bottom:8px}
.footer p{font-size:12px;color:var(--mut);line-height:1.9}
.footer p strong{color:var(--txt)}
.gen-note{grid-column:1/-1;font-size:12px;color:var(--mut);padding-top:14px;
  border-top:1px solid var(--brd2);margin-top:4px}
code{font-family:var(--mono);background:rgba(255,255,255,.08);
  padding:1px 6px;border-radius:4px;font-size:12px}
</style>
</head>
<body>

<div class="hdr">
  <div>
    <h1>ASX RTO Screener</h1>
    <div class="sub">Abnormal volume &amp; price drift — Materials &amp; Energy</div>
  </div>
  <div class="hdr-right">
    <div>Last run: <strong id="ts"></strong></div>
    <div id="run-sub" style="margin-top:3px"></div>
  </div>
</div>

<div class="stats" id="stats"></div>

<div class="section">
  <div class="sec-hdr t1" id="hdr1" onclick="tog(1)">
    <h2>Tier 1 — Unexplained Signals</h2>
    <span class="badge" id="b1"></span>
    <span class="note">No qualifying announcement found in last ~30 trading days</span>
    <span class="chev" id="chev1">▾</span>
  </div>
  <div class="sec-body" id="body1">
    <div class="tbl-wrap"><table id="tbl1">
      <thead><tr>
        <th onclick="srt(1,'ticker')">Ticker</th>
        <th onclick="srt(1,'name')">Company</th>
        <th class="r" onclick="srt(1,'vol_ratio')">Vol Ratio</th>
        <th class="r" onclick="srt(1,'drift_pct')">Drift %</th>
        <th class="r" onclick="srt(1,'ret_pct')">Actual Ret %</th>
        <th onclick="srt(1,'coverage')">API Coverage</th>
        <th class="r" onclick="srt(1,'score')">Score</th>
      </tr></thead>
      <tbody id="tb1"></tbody>
    </table></div>
  </div>
</div>

<div class="section">
  <div class="sec-hdr t2" id="hdr2" onclick="tog(2)">
    <h2>Tier 2 — Unclear</h2>
    <span class="badge" id="b2"></span>
    <span class="note">Markit API history &lt; 45 days — cannot confirm or deny announcement</span>
    <span class="chev" id="chev2">▾</span>
  </div>
  <div class="sec-body" id="body2">
    <div class="tbl-wrap"><table id="tbl2">
      <thead><tr>
        <th onclick="srt(2,'ticker')">Ticker</th>
        <th onclick="srt(2,'name')">Company</th>
        <th class="r" onclick="srt(2,'vol_ratio')">Vol Ratio</th>
        <th class="r" onclick="srt(2,'drift_pct')">Drift %</th>
        <th class="r" onclick="srt(2,'ret_pct')">Actual Ret %</th>
        <th onclick="srt(2,'coverage')">API Coverage</th>
        <th class="r" onclick="srt(2,'score')">Score</th>
      </tr></thead>
      <tbody id="tb2"></tbody>
    </table></div>
  </div>
</div>

<div class="section">
  <div class="sec-hdr t3 shut" id="hdr3" onclick="tog(3)">
    <h2>Tier 3 — Explained</h2>
    <span class="badge" id="b3"></span>
    <span class="note">Material public announcement found — move likely legitimate</span>
    <span class="chev" id="chev3">▾</span>
  </div>
  <div class="sec-body" id="body3" style="display:none">
    <div class="tbl-wrap"><table id="tbl3">
      <thead><tr>
        <th onclick="srt(3,'ticker')">Ticker</th>
        <th onclick="srt(3,'name')">Company</th>
        <th class="r" onclick="srt(3,'vol_ratio')">Vol Ratio</th>
        <th class="r" onclick="srt(3,'drift_pct')">Drift %</th>
        <th class="r" onclick="srt(3,'ret_pct')">Actual Ret %</th>
        <th onclick="srt(3,'expl_type')">Explanation</th>
        <th class="r" onclick="srt(3,'score')">Score</th>
      </tr></thead>
      <tbody id="tb3"></tbody>
    </table></div>
  </div>
</div>

<div class="footer">
  <div>
    <h3>Signal Logic</h3>
    <p>
      Baseline window: T&#8722;90 to T&#8722;31 &nbsp;(60 trading days, OLS on ^AXSO)<br>
      Signal window: T&#8722;30 to T&#8722;1 &nbsp;(30 trading days)<br>
      <strong>Flag</strong> = vol_ratio &#8805; 2&#215; <em>and</em> cum. abnormal drift &#8805; +5%<br>
      Score = 0.4&#215;z(vol) + 0.6&#215;z(drift), ranked within tier
    </p>
  </div>
  <div>
    <h3>Tier Definitions</h3>
    <p>
      <strong style="color:var(--red)">Tier 1 (Unexplained)</strong> &mdash; no placement,
      entitlement offer, capital raise, trading halt, drilling result,
      resource update, M&amp;A announcement, or ASX query in signal window.<br>
      <strong style="color:var(--amb)">Tier 2 (Unclear)</strong> &mdash; Markit API
      returned &lt;45 days of history; cannot confirm.<br>
      <strong style="color:var(--grn)">Tier 3 (Explained)</strong> &mdash; qualifying
      announcement found; move is likely legitimate.
    </p>
  </div>
  <div class="gen-note" id="gen-note"></div>
</div>

<script>
const D = __DATA_JSON__;

function pct(n, dec) {
  if (n == null || isNaN(n)) return '—';
  return (n > 0 ? '+' : '') + n.toFixed(dec) + '%';
}
function cc(n) { return n > 0 ? 'pos' : n < 0 ? 'neg' : 'dim'; }

document.getElementById('ts').textContent = D.run_ts;
document.getElementById('run-sub').textContent =
  D.screened + ' companies screened · ' + D.flagged + ' flagged';

// Stats chips
const chips = [
  ['chip-r', D.n1, 'Unexplained'],
  ['chip-a', D.n2, 'Unclear'],
  ['chip-g', D.n3, 'Explained'],
  ['', D.screened, 'screened'],
  ['', D.flagged,  'flagged'],
];
document.getElementById('stats').innerHTML = chips.map(([cls, n, lbl]) =>
  `<span class="chip ${cls}"><span class="n">${n}</span> ${lbl}</span>`
).join('');

['b1','b2','b3'].forEach((id, i) => {
  document.getElementById(id).textContent = D['n' + (i+1)];
});

document.getElementById('gen-note').innerHTML =
  'Generated by screener.py &mdash; run <code>python3 screener.py</code> to update &mdash; ' + D.run_ts;

// ── Table rendering ──────────────────────────────────────────────────────────
function render(tier) {
  const rows = D['t' + tier] || [];
  const tb   = document.getElementById('tb' + tier);
  const expl = tier === 3;
  if (!rows.length) {
    tb.innerHTML = '<tr><td colspan="7" class="empty">No signals in this tier</td></tr>';
    return;
  }
  tb.innerHTML = rows.map(r => {
    const sixthCell = expl
      ? `<td><span class="extyp">${r.expl_type || '?'}</span>` +
        `<span class="exhl" title="${r.expl_hl}">${r.exhl_short || r.expl_hl || '—'}</span></td>`
      : `<td><span class="cov">${r.coverage || '—'}</span></td>`;
    return `<tr>
      <td class="tkr"><a href="${r.url}" target="_blank">${r.ticker}</a></td>
      <td class="nm" title="${r.name}">${r.name}</td>
      <td class="r dim">${r.vol_ratio.toFixed(2)}&times;</td>
      <td class="r bold ${cc(r.drift_pct)}">${pct(r.drift_pct,1)}</td>
      <td class="r ${cc(r.ret_pct)}">${pct(r.ret_pct,1)}</td>
      ${sixthCell}
      <td class="r dim">${r.score.toFixed(3)}</td>
    </tr>`;
  }).join('');
}
[1,2,3].forEach(render);

// ── Sort ─────────────────────────────────────────────────────────────────────
let SS = {};
function srt(tier, key) {
  const data = D['t' + tier];
  const s    = SS[tier] || {};
  const asc  = s.key === key ? !s.asc : false;
  data.sort((a, b) => {
    const av = a[key], bv = b[key];
    if (av === bv) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    return (av < bv ? -1 : 1) * (asc ? 1 : -1);
  });
  SS[tier] = {key, asc};
  render(tier);
  document.querySelectorAll('#tbl' + tier + ' thead th').forEach(th => {
    th.classList.remove('asc','desc');
    const oc = th.getAttribute('onclick') || '';
    if (oc.includes("'" + key + "'")) th.classList.add(asc ? 'asc' : 'desc');
  });
}

// ── Toggle sections ──────────────────────────────────────────────────────────
function tog(t) {
  const body = document.getElementById('body' + t);
  const hdr  = document.getElementById('hdr'  + t);
  const chev = document.getElementById('chev' + t);
  const open = body.style.display === 'none';
  body.style.display = open ? '' : 'none';
  hdr.classList.toggle('shut', !open);
  chev.textContent = open ? '▾' : '▸';
}
</script>
</body>
</html>'''


if __name__ == "__main__":
    main()
