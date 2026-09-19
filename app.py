import base64
import json
import math
import re
from datetime import datetime
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Reverse Split Scanner", page_icon="🔎", layout="wide")

HEADERS = {
    "User-Agent": "ReverseSplitScanner/3.0 research-app",
    "Accept": "application/json,text/html,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YAHOO_SEARCH = "https://query1.finance.yahoo.com/v1/finance/search"
YAHOO_SUMMARY = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
NASDAQ_NONCOMPLIANT = "https://www.nasdaq.com/market-activity/stocks/non-compliant-company-list"
IBD_URL = "https://www.iborrowdesk.com/report/{ticker}"
IBD_API_CANDIDATES = [
    "https://www.iborrowdesk.com/api/ticker/{ticker}",
    "https://www.iborrowdesk.com/api/report/{ticker}",
]

BASE_LISTS = {
    "Favorites": "favorites",
    "Waiting for Price Drop": "price_drop",
    "Waiting for Short Drop": "short_drop",
    "Waiting for RSI Drop": "rsi_drop",
    "Waiting for Support / Price Stability": "support",
    "Negative Stocks": "negative",
    "Short Drop + Support": "short_support",
    "Price Drop + RSI Drop": "price_rsi",
    "Price Drop + Support": "price_support",
    "Short Drop + RSI Drop": "short_rsi",
    "Price Drop + Short Drop + Support": "price_short_support",
}


def clean_num(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_human_num(value):
    if value is None:
        return np.nan
    s = str(value).strip().upper().replace(",", "")
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*([KMBT]?)", s)
    if not m:
        return np.nan
    return float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[m.group(2)]


def yahoo_get(ticker, params=None):
    r = requests.get(YAHOO_CHART.format(ticker=ticker.upper()), params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600)
def get_price_data(ticker):
    payload = yahoo_get(
        ticker,
        {"period1": 0, "period2": int(datetime.now().timestamp()), "interval": "1d", "events": "div,splits"},
    )
    result = payload.get("chart", {}).get("result")
    if not result:
        return pd.DataFrame(), {}
    result = result[0]
    timestamps = result.get("timestamp", [])
    q = result.get("indicators", {}).get("quote", [{}])[0]
    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
        "open": q.get("open", []), "high": q.get("high", []), "low": q.get("low", []),
        "close": q.get("close", []), "volume": q.get("volume", []),
    })
    if not df.empty:
        df["date"] = df["date"].dt.normalize()
        df = df.dropna(subset=["close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    splits = []
    for item in (result.get("events", {}) or {}).get("splits", {}).values():
        ts = item.get("date")
        if ts is not None:
            splits.append({"date": pd.to_datetime(ts, unit="s").normalize(), "numerator": item.get("numerator"), "denominator": item.get("denominator")})
    return df, {"splits": pd.DataFrame(splits)}


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.where(~((avg_loss == 0) & (avg_gain > 0)), 100).where(~((avg_gain == 0) & (avg_loss > 0)), 0)


def reverse_splits(split_df):
    if split_df.empty:
        return split_df
    x = split_df.copy()
    x["numerator"] = pd.to_numeric(x["numerator"], errors="coerce")
    x["denominator"] = pd.to_numeric(x["denominator"], errors="coerce")
    return x[(x["numerator"] < x["denominator"]) & x["numerator"].notna() & x["denominator"].notna()].sort_values("date")


@st.cache_data(ttl=1800)
def sec_ticker_map():
    try:
        r = requests.get(SEC_TICKERS, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        return {str(row.get("ticker", "")).upper(): {"cik": str(row.get("cik_str", "")).zfill(10), "title": row.get("title", "")} for row in data.values() if row.get("ticker")}
    except Exception:
        return {}


def sec_json(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600)
def get_sec_profile(ticker):
    item = sec_ticker_map().get(ticker.upper())
    if not item:
        return {}
    cik = item["cik"]
    try:
        sub = sec_json(SEC_SUBMISSIONS.format(cik=cik))
    except Exception:
        sub = {}
    try:
        facts = sec_json(SEC_FACTS.format(cik=cik))
    except Exception:
        facts = {}
    country = None
    for key in ("business", "mailing"):
        addr = (sub.get("addresses", {}) or {}).get(key, {})
        country = addr.get("country") or country
        if country:
            break
    if not country and sub.get("stateOfIncorporation"):
        country = "United States"
    shares = np.nan
    facts_obj = facts.get("facts", {}) if isinstance(facts, dict) else {}
    for namespace, tag in [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")]:
        units = facts_obj.get(namespace, {}).get(tag, {}).get("units", {})
        for arr in units.values():
            candidates = []
            for row in arr:
                val = clean_num(row.get("val"))
                if np.isfinite(val):
                    filed = pd.to_datetime(row.get("filed"), errors="coerce")
                    candidates.append((filed, val))
            if candidates:
                candidates.sort(key=lambda x: (pd.isna(x[0]), x[0]))
                shares = candidates[-1][1]
                break
        if np.isfinite(shares):
            break
    return {"country": country, "shares": shares, "cik": cik, "title": item.get("title"), "submissions": sub}


@st.cache_data(ttl=900)
def get_yahoo_profile(ticker):
    meta = {}
    try:
        payload = yahoo_get(ticker, {"range": "5d", "interval": "1d"})
        result = payload.get("chart", {}).get("result")
        meta = result[0].get("meta", {}) if result else {}
    except Exception:
        pass
    # Best-effort quoteSummary fallback for float shares and other share statistics.
    try:
        r = requests.get(
            YAHOO_SUMMARY.format(ticker=ticker.upper()),
            params={"modules": "defaultKeyStatistics,price,summaryDetail,calendarEvents"},
            headers=HEADERS, timeout=20,
        )
        r.raise_for_status()
        result = r.json().get("quoteSummary", {}).get("result")
        if result:
            blob = result[0]
            stats = blob.get("defaultKeyStatistics", {})
            summary = blob.get("summaryDetail", {})
            for src in (stats, summary):
                for key in ("sharesOutstanding", "floatShares"):
                    val = src.get(key)
                    if isinstance(val, dict):
                        val = val.get("raw")
                    if val is not None:
                        meta[key] = val
            cal = blob.get("calendarEvents", {})
            earnings = cal.get("earnings", {}).get("earningsDate", []) if isinstance(cal, dict) else []
            if earnings:
                meta["earningsDate"] = earnings[0].get("raw") if isinstance(earnings[0], dict) else earnings[0]
    except Exception:
        pass
    return meta


def get_profile(ticker):
    sec = get_sec_profile(ticker)
    yh = get_yahoo_profile(ticker)
    shares = sec.get("shares", np.nan)
    if not np.isfinite(shares):
        shares = clean_num(yh.get("sharesOutstanding"))
    float_shares = clean_num(yh.get("floatShares"))
    country = sec.get("country") or "N/A"
    return {"shares": shares, "float_shares": float_shares, "country": country, "title": sec.get("title") or ticker.upper(), "sec": sec, "yh": yh}


@st.cache_data(ttl=900)
def get_iborrowdesk(ticker):
    ticker = ticker.upper().strip()
    browser_headers = {**HEADERS, "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/128 Safari/537.36", "Referer": IBD_URL.format(ticker=ticker)}
    for endpoint in IBD_API_CANDIDATES:
        try:
            r = requests.get(endpoint.format(ticker=ticker), headers=browser_headers, timeout=20)
            r.raise_for_status()
            payload = r.json()
            candidates = []
            def find_rows(obj):
                found = []
                if isinstance(obj, list):
                    if obj and all(isinstance(x, dict) for x in obj[: min(5, len(obj))]):
                        keys = set().union(*(x.keys() for x in obj[: min(10, len(obj))]))
                        if any(k in keys for k in ("fee", "borrow_fee")) and any(k in keys for k in ("available", "shares_available")):
                            return obj
                    for item in obj:
                        found = find_rows(item)
                        if found: return found
                elif isinstance(obj, dict):
                    for value in obj.values():
                        found = find_rows(value)
                        if found: return found
                return []
            candidates = find_rows(payload)
            if candidates:
                rows = []
                for row in candidates:
                    if not isinstance(row, dict):
                        continue
                    dt = pd.to_datetime(row.get("date") or row.get("reported") or row.get("timestamp"), errors="coerce")
                    if pd.isna(dt):
                        continue
                    rows.append({"reported": dt, "fee": clean_num(row.get("fee") or row.get("borrow_fee")), "available": parse_human_num(row.get("available") or row.get("shares_available"))})
                if rows:
                    t = pd.DataFrame(rows).drop_duplicates("reported").sort_values("reported")
                    cutoff = pd.Timestamp.now() - pd.Timedelta(days=31)
                    return t[t["reported"] >= cutoff].reset_index(drop=True)
        except Exception:
            continue
    try:
        r = requests.get(IBD_URL.format(ticker=ticker), headers=browser_headers, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception:
        return pd.DataFrame(columns=["reported", "fee", "available"])
    try:
        tables = pd.read_html(html)
    except Exception:
        tables = []
    for table in tables:
        cols = [str(c).strip().lower() for c in table.columns]
        if "reported" in cols and any("fee" in c for c in cols) and any("available" in c for c in cols):
            ren = {}
            for c in table.columns:
                lc = str(c).lower()
                if lc == "reported": ren[c] = "reported"
                elif "fee" in lc: ren[c] = "fee"
                elif "available" in lc: ren[c] = "available"
            t = table.rename(columns=ren)
            t["reported"] = pd.to_datetime(t["reported"], errors="coerce")
            t["fee"] = t["fee"].map(clean_num)
            t["available"] = t["available"].map(parse_human_num)
            t = t.dropna(subset=["reported"]).sort_values("reported")
            return t[t["reported"] >= pd.Timestamp.now() - pd.Timedelta(days=31)].reset_index(drop=True)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    pat = re.compile(r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}\s+(?:AM|PM))\s+([-+]?\d+(?:\.\d+)?)\s*%\s+([0-9.,]+(?:[KMBT])?)", re.I)
    rows = [{"reported": pd.to_datetime(m.group(1)), "fee": float(m.group(2)), "available": parse_human_num(m.group(3))} for m in pat.finditer(text)]
    t = pd.DataFrame(rows)
    if t.empty:
        return pd.DataFrame(columns=["reported", "fee", "available"])
    return t.drop_duplicates("reported").sort_values("reported").loc[lambda x: x["reported"] >= pd.Timestamp.now() - pd.Timedelta(days=31)].reset_index(drop=True)


@st.cache_data(ttl=1800)
def get_share_change(ticker):
    prof = get_sec_profile(ticker)
    facts = {}
    try:
        facts = sec_json(SEC_FACTS.format(cik=prof.get("cik"))) if prof.get("cik") else {}
    except Exception:
        return {"latest": np.nan, "previous": np.nan, "change_pct": np.nan, "date": None}
    arr = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    rows = []
    for row in arr:
        val = clean_num(row.get("val")); dt = pd.to_datetime(row.get("filed"), errors="coerce")
        if np.isfinite(val) and pd.notna(dt): rows.append((dt, val))
    if not rows: return {"latest": np.nan, "previous": np.nan, "change_pct": np.nan, "date": None}
    # Keep the latest reported value for each filing date, then compare two latest distinct dates.
    x = pd.DataFrame(rows, columns=["date", "value"]).sort_values("date").drop_duplicates("date", keep="last")
    latest = x.iloc[-1]; previous = x.iloc[-2] if len(x) > 1 else None
    change = np.nan if previous is None or previous.value == 0 else (latest.value / previous.value - 1) * 100
    return {"latest": latest.value, "previous": previous.value if previous is not None else np.nan, "change_pct": change, "date": latest.date}


@st.cache_data(ttl=1800)
def get_sec_news(ticker):
    prof = get_sec_profile(ticker)
    sub = prof.get("submissions", {})
    recent = sub.get("filings", {}).get("recent", {}) if isinstance(sub, dict) else {}
    rows = []
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    for i, form in enumerate(forms[:80]):
        if form not in {"8-K", "8-K/A", "S-1", "S-1/A", "S-3", "S-3/A", "424B3", "424B5", "10-Q", "10-Q/A", "10-K", "10-K/A", "25", "25-NSE", "DEF 14A"}:
            continue
        dt = pd.to_datetime(dates[i], errors="coerce")
        if pd.isna(dt) or dt < pd.Timestamp.now().normalize() - pd.Timedelta(days=45):
            continue
        accession = str(accessions[i]).replace("-", "")
        doc = docs[i] if i < len(docs) else ""
        url = f"https://www.sec.gov/Archives/edgar/data/{int(prof['cik'])}/{accession}/{doc}" if doc else "https://www.sec.gov/edgar/browse/?CIK=" + str(int(prof["cik"]))
        desc = {"8-K": "Material company filing", "S-1": "Registration / potential financing", "S-3": "Shelf registration / financing capacity", "10-Q": "Quarterly report", "10-K": "Annual report", "25": "Delisting filing", "25-NSE": "Nasdaq suspension/delisting filing", "DEF 14A": "Proxy filing"}.get(form, form)
        rows.append({"date": dt, "event": desc, "type": form, "source": "SEC", "url": url})
    return pd.DataFrame(rows).drop_duplicates("url") if rows else pd.DataFrame(columns=["date", "event", "type", "source", "url"])


@st.cache_data(ttl=1800)
def get_yahoo_news(ticker):
    try:
        r = requests.get(YAHOO_SEARCH, params={"q": ticker, "quotesCount": 5, "newsCount": 10, "enableFuzzyQuery": "false"}, headers=HEADERS, timeout=20)
        r.raise_for_status()
        news = r.json().get("news", [])
        rows = []
        for item in news:
            ts = pd.to_datetime(item.get("providerPublishTime"), unit="s", errors="coerce")
            if pd.isna(ts) or ts < pd.Timestamp.now(tz=None) - pd.Timedelta(days=14):
                continue
            rows.append({"date": ts, "event": item.get("title", "News"), "type": "News", "source": item.get("publisher", "Yahoo Finance"), "url": item.get("link", "")})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["date", "event", "type", "source", "url"])


@st.cache_data(ttl=1800)
def get_nasdaq_status(ticker):
    try:
        r = requests.get(NASDAQ_NONCOMPLIANT, headers={**HEADERS, "User-Agent": "Mozilla/5.0 Chrome/128 Safari/537.36"}, timeout=20)
        r.raise_for_status()
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text)).upper()
        # Dynamic Nasdaq pages may not expose rows to a plain request. Only return a status when the ticker is explicitly present.
        if re.search(rf"\b{re.escape(ticker.upper())}\b", text):
            return {"status": "Non-compliant / delisting list match", "source": "Nasdaq", "url": NASDAQ_NONCOMPLIANT}
    except Exception:
        pass
    return None


def classify_news(row):
    text = (str(row.get("event", "")) + " " + str(row.get("type", ""))).lower()
    negative = ["dilution", "offering", "registration", "delisting", "non-compliance", "noncompliance", "suspension", "bankruptcy", "going concern", "restatement", "investigation"]
    positive = ["compliance regained", "approval", "contract", "agreement", "partnership", "acquisition", "award", "milestone"]
    if any(k in text for k in negative): return "Negative"
    if any(k in text for k in positive): return "Positive"
    return "Neutral"


def expected_event_type(event):
    text = str(event).lower()
    if any(k in text for k in ["offering", "registration", "dilution", "non-compliance", "delisting"]):
        return "Negative / risk"
    if any(k in text for k in ["approval", "compliance regained", "contract", "partnership"]):
        return "Positive / catalyst"
    return "Neutral / watch"


def get_news_bundle(ticker):
    sec = get_sec_news(ticker)
    yh = get_yahoo_news(ticker)
    nasdaq = get_nasdaq_status(ticker)
    frames = [x for x in (sec, yh) if not x.empty]
    news = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "event", "type", "source", "url"])
    if not news.empty:
        news["date"] = pd.to_datetime(news["date"], errors="coerce")
        news["label"] = news.apply(classify_news, axis=1)
        news = news.sort_values("date", ascending=False).drop_duplicates("event").head(12)
    return news, nasdaq


def encode_lists(lists):
    raw = json.dumps({k: sorted(set(v)) for k, v in lists.items()}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode_lists(value):
    if not value:
        return {v: [] for v in BASE_LISTS.values()}
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()
        data = json.loads(raw)
        return {k: list(map(str.upper, data.get(k, []))) for k in BASE_LISTS.values()}
    except Exception:
        return {v: [] for v in BASE_LISTS.values()}


def load_lists():
    return decode_lists(st.query_params.get("lists", ""))


def save_lists(lists):
    st.query_params["lists"] = encode_lists(lists)


def add_to_list(lists, key, ticker):
    if ticker not in lists[key]:
        lists[key].append(ticker)
        save_lists(lists)
        st.rerun()


def remove_from_list(lists, key, ticker):
    lists[key] = [x for x in lists[key] if x != ticker]
    save_lists(lists)
    st.rerun()


def fmt_shares(v):
    return "N/A" if v is None or not np.isfinite(v) else f"{v:,.0f}"


def inject_css():
    st.markdown("""
    <style>
    .stApp { background:#06111f; }
    .block-container { max-width:1450px; padding-top:1.1rem; padding-bottom:2rem; }
    .hero,.panel,.card { border:1px solid #173556; background:#07182c; border-radius:16px; box-shadow:0 8px 28px rgba(0,0,0,.14); }
    .hero { padding:18px 22px; margin-bottom:14px; }
    .hero-title { color:#f5f8ff; font-size:30px; font-weight:800; }
    .hero-sub,.small-note { color:#91a5bf; font-size:13px; }
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#102b4c; color:#74b5ff; font-size:12px; margin-left:8px; }
    .card { padding:15px 17px; min-height:95px; }
    .card-label { color:#91a5bf; font-size:12px; margin-bottom:8px; }
    .card-value { color:#f4f7fb; font-size:23px; font-weight:800; }
    .section { color:#f4f7fb; font-size:20px; font-weight:800; margin:18px 0 10px; }
    .panel { padding:16px; }
    .tag { display:inline-block; padding:3px 8px; border-radius:999px; background:#12263d; color:#b8c8da; font-size:11px; margin-right:5px; }
    .green { color:#59e29a; } .red { color:#ff6969; } .yellow { color:#f6c85f; }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label, value):
    return f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div></div>'


def list_manager(ticker, lists):
    st.markdown('<div class="section">My Lists</div>', unsafe_allow_html=True)
    options = ["Favorites", "Waiting for Price Drop", "Waiting for Short Drop", "Waiting for RSI Drop", "Waiting for Support / Price Stability", "Negative Stocks"]
    with st.expander("Add this stock to lists", expanded=False):
        cols = st.columns(2)
        for i, name in enumerate(options):
            key = BASE_LISTS[name]
            with cols[i % 2]:
                if ticker in lists[key]:
                    if st.button(f"✓ {name}", key=f"remove_{key}_{ticker}", use_container_width=True): remove_from_list(lists, key, ticker)
                else:
                    if st.button(f"+ {name}", key=f"add_{key}_{ticker}", use_container_width=True): add_to_list(lists, key, ticker)
    # Combined lists are automatic intersections of the base lists.
    combinations = {
        "Short Drop + Support": ["short_drop", "support"],
        "Price Drop + RSI Drop": ["price_drop", "rsi_drop"],
        "Price Drop + Support": ["price_drop", "support"],
        "Short Drop + RSI Drop": ["short_drop", "rsi_drop"],
        "Price Drop + Short Drop + Support": ["price_drop", "short_drop", "support"],
    }
    st.caption("Combined lists are automatic: a stock appears when it is present in all of the selected base lists.")
    rows = []
    for name, keys in combinations.items():
        common = set(lists[keys[0]])
        for key in keys[1:]: common &= set(lists[key])
        rows.append({"List": name, "Stocks": ", ".join(sorted(common)) if common else "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=220)
    with st.expander("Saved lists", expanded=False):
        for name in options:
            vals = lists[BASE_LISTS[name]]
            st.markdown(f"**{name}** — {', '.join(vals) if vals else 'Empty'}")


def main():
    inject_css()
    lists = load_lists()
    top1, top2 = st.columns([6, 2])
    with top1:
        st.markdown('<div class="hero"><div class="hero-title">🔎 Reverse Split Scanner</div><div class="hero-sub">Simple reverse-split research: price, short data, shares and only the most important events.</div></div>', unsafe_allow_html=True)
    with top2:
        ticker = st.text_input("Ticker", value=st.query_params.get("ticker", "GNPX").upper(), label_visibility="collapsed", placeholder="Ticker", key="ticker_input").strip().upper()
        if ticker:
            st.query_params["ticker"] = ticker
    if not ticker:
        st.stop()

    with st.spinner("Loading market and filing data..."):
        try:
            prices, event_data = get_price_data(ticker)
        except Exception as e:
            st.error(f"Could not load market data: {e}")
            st.stop()
    if prices.empty:
        st.error("No price data found for this ticker.")
        st.stop()

    splits = reverse_splits(event_data.get("splits", pd.DataFrame()))
    split_date = None; split_ratio = "N/A"; split_high = np.nan
    if not splits.empty:
        selected = splits.iloc[-1]
        split_date = pd.Timestamp(selected["date"]).normalize()
        split_ratio = f"{int(selected['numerator'])}:{int(selected['denominator'])}"
        day = prices[prices["date"] == split_date]
        if day.empty:
            day = prices[prices["date"].between(split_date - pd.Timedelta(days=3), split_date + pd.Timedelta(days=3))].head(1)
        if not day.empty: split_high = float(day.iloc[0]["high"])
    prices["rsi14"] = rsi(prices["close"], 14)
    latest_close = float(prices.iloc[-1]["close"])
    latest_rsi = float(prices.iloc[-1]["rsi14"]) if pd.notna(prices.iloc[-1]["rsi14"]) else np.nan
    drop = (latest_close / split_high - 1) * 100 if np.isfinite(split_high) and split_high else np.nan
    profile = get_profile(ticker)
    ibd = get_iborrowdesk(ticker)
    ibd_latest = ibd.iloc[-1] if not ibd.empty else None
    news, nasdaq = get_news_bundle(ticker)

    title = profile["title"] or ticker
    st.markdown(f'<div class="hero"><div class="hero-title">{title} <span class="pill">{ticker}</span></div><div class="hero-sub">Reverse split analysis · Updated {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</div></div>', unsafe_allow_html=True)

    if nasdaq:
        st.warning(f"⚠️ Nasdaq: {nasdaq['status']} — source: Nasdaq")

    values = [
        ("Reverse Split Date", split_date.strftime("%Y-%m-%d") if split_date is not None else "N/A"),
        ("Reverse Split Ratio", split_ratio),
        ("High on Split Day", f"${split_high:,.2f}" if np.isfinite(split_high) else "N/A"),
        ("Current / Latest Close", f"${latest_close:,.2f}"),
        ("Drop from Split-Day High", f"{drop:.1f}%" if np.isfinite(drop) else "N/A"),
        ("Daily RSI (14)", f"{latest_rsi:.1f}" if np.isfinite(latest_rsi) else "N/A"),
        ("Shares Outstanding", fmt_shares(profile["shares"])),
        ("Company Country", profile["country"]),
    ]
    for row in (values[:4], values[4:]):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, row):
            with col: st.markdown(metric_card(label, value), unsafe_allow_html=True)

    # Shares section
    share_change = get_share_change(ticker)
    float_shares = profile.get("float_shares", np.nan)
    float_pct = (float_shares / profile["shares"] * 100) if np.isfinite(float_shares) and np.isfinite(profile["shares"]) and profile["shares"] else np.nan
    st.markdown('<div class="section">Shares</div>', unsafe_allow_html=True)
    shares_df = pd.DataFrame([
        {"Metric": "Shares Outstanding", "Value": fmt_shares(profile["shares"]), "Source": "SEC EDGAR when available"},
        {"Metric": "Previous Reported Shares", "Value": fmt_shares(share_change["previous"]), "Source": "SEC EDGAR"},
        {"Metric": "Reported Share Change", "Value": f"{share_change["change_pct"]:+.1f}%" if np.isfinite(share_change["change_pct"]) else "N/A", "Source": "SEC EDGAR"},
        {"Metric": "Free Float", "Value": fmt_shares(float_shares), "Source": "Yahoo Finance when available"},
        {"Metric": "Free Float %", "Value": f"{float_pct:.1f}%" if np.isfinite(float_pct) else "N/A", "Source": "Calculated when both values are available"},
    ])
    st.dataframe(shares_df, hide_index=True, use_container_width=True, height=220)

    # Short data: full month table
    st.markdown('<div class="section">Short Data — Last 1 Month</div>', unsafe_allow_html=True)
    if ibd.empty:
        st.info("No IBorrowDesk 1M history could be read for this ticker.")
    else:
        short_df = ibd.sort_values("reported", ascending=False).copy()
        short_df["Reported"] = pd.to_datetime(short_df["reported"]).dt.strftime("%Y-%m-%d %H:%M")
        short_df = short_df.rename(columns={"fee": "Borrow Fee", "available": "Shares Available"})[["Reported", "Borrow Fee", "Shares Available"]]
        st.dataframe(short_df, hide_index=True, use_container_width=True, height=460, column_config={"Borrow Fee": st.column_config.NumberColumn("Borrow Fee", format="%.2f%%"), "Shares Available": st.column_config.NumberColumn("Shares Available", format="%d")})
        st.caption(f"Showing all available IBorrowDesk readings in the last 31 days: {len(short_df)} rows. IBorrowDesk publishes reported fee/availability readings and supports a 1M view. ")

    # Price table only, no charts
    st.markdown('<div class="section">Last 30 Trading Days</div>', unsafe_allow_html=True)
    last30 = prices.tail(30).copy().sort_values("date", ascending=False)
    last30["Change %"] = last30["close"].pct_change(periods=-1) * 100
    display = last30[["date", "open", "high", "low", "close", "volume", "Change %"]].copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    display = display.rename(columns={"date":"Date", "open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume"})
    st.dataframe(display, hide_index=True, use_container_width=True, height=540, column_config={"Open": st.column_config.NumberColumn("Open", format="$%.2f"), "High": st.column_config.NumberColumn("High", format="$%.2f"), "Low": st.column_config.NumberColumn("Low", format="$%.2f"), "Close": st.column_config.NumberColumn("Close", format="$%.2f"), "Volume": st.column_config.NumberColumn("Volume", format="%d"), "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%")})

    # Lists
    list_manager(ticker, lists)

    # News
    st.markdown('<div class="section">Important News</div>', unsafe_allow_html=True)
    if news.empty:
        st.info("No recent major news or SEC events were found from the available public feeds.")
    else:
        show = news.head(3).copy()
        show["Date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
        show["Event"] = show["event"].str.slice(0, 115)
        show["Type"] = show["label"]
        show["Source"] = show["source"]
        st.dataframe(show[["Date", "Event", "Type", "Source"]], hide_index=True, use_container_width=True, height=190)
        for _, row in show.iterrows():
            if row.get("url"):
                st.markdown(f"- [{row['event']}]({row['url']}) — {row['source']}")

    # Upcoming / expected, only documented events; not price predictions
    st.markdown('<div class="section">Upcoming / Expected Events</div>', unsafe_allow_html=True)
    upcoming = []
    earnings_raw = profile.get("yh", {}).get("earningsDate")
    if earnings_raw:
        try:
            earnings_dt = pd.to_datetime(earnings_raw, unit="s", errors="coerce")
            if pd.isna(earnings_dt): earnings_dt = pd.to_datetime(earnings_raw, errors="coerce")
            if pd.notna(earnings_dt) and earnings_dt >= pd.Timestamp.now().normalize():
                upcoming.append({"Event": "Earnings report", "Status": "Scheduled / expected date", "Source": "Yahoo Finance calendar", "Date": earnings_dt.strftime("%Y-%m-%d")})
        except Exception:
            pass
    sub = profile["sec"].get("submissions", {}) if profile.get("sec") else {}
    recent = sub.get("filings", {}).get("recent", {}) if isinstance(sub, dict) else {}
    forms = recent.get("form", []); dates = recent.get("filingDate", [])
    for i, form in enumerate(forms[:40]):
        if form in {"10-Q", "10-K"}:
            dt = pd.to_datetime(dates[i], errors="coerce")
            if pd.notna(dt): upcoming.append({"Event": "Next financial reporting cycle", "Status": "Expected / filing cycle", "Source": "SEC", "Date": "Not fixed by SEC filing history"}); break
    if nasdaq:
        upcoming.append({"Event": "Nasdaq compliance / listing status", "Status": "Monitor official Nasdaq updates", "Source": "Nasdaq", "Date": "Ongoing"})
    if upcoming:
        u = pd.DataFrame(upcoming).head(3)
        u["Expected Type"] = u["Event"].map(expected_event_type)
        st.dataframe(u[["Event", "Expected Type", "Status", "Source", "Date"]], hide_index=True, use_container_width=True, height=170)
    else:
        st.caption("No dated upcoming event was found in the available public filings. This section does not predict price direction.")

    st.markdown('<div class="section">Sources</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Market price/splits: Yahoo Finance chart feed. Company filings/shares: SEC EDGAR. Short availability: IBorrowDesk public data. News discovery: Yahoo Finance search feed. Nasdaq status: Nasdaq non-compliant company list when publicly exposed. News labels are simple event classifications, not investment recommendations.</div>', unsafe_allow_html=True)
    st.caption("Research/education only. No buy/sell recommendation is provided.")


if __name__ == "__main__":
    main()
