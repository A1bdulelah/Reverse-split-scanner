import re
import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Reverse Split Scanner", layout="wide")

HEADERS = {
    "User-Agent": "ReverseSplitScanner/2.0 contact@example.com",
    "Accept": "application/json,text/html,text/plain,*/*",
}

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YAHOO_QUOTE = "https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
IBD_URL = "https://www.iborrowdesk.com/report/{ticker}"
IBD_API_URL = "https://www.iborrowdesk.com/api/ticker/{ticker}"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def clean_num(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("$", "")
    s = s.replace("%", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def yahoo_get(ticker, params=None):
    r = requests.get(
        YAHOO_CHART.format(ticker=ticker.upper()),
        params=params,
        headers=HEADERS,
        timeout=20,
    )
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
        "open": q.get("open", []),
        "high": q.get("high", []),
        "low": q.get("low", []),
        "close": q.get("close", []),
        "volume": q.get("volume", []),
    })
    if not df.empty:
        df["date"] = df["date"].dt.normalize()
        df = df.dropna(subset=["close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)

    events = result.get("events", {}) or {}
    splits = events.get("splits", {}) or {}
    split_rows = []
    for _, item in splits.items():
        ts = item.get("date")
        if ts is None:
            continue
        split_rows.append({
            "date": pd.to_datetime(ts, unit="s").normalize(),
            "numerator": item.get("numerator"),
            "denominator": item.get("denominator"),
        })
    split_df = pd.DataFrame(split_rows)
    return df, {"splits": split_df}


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100)
    out = out.where(~((avg_gain == 0) & (avg_loss > 0)), 0)
    return out


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
        rows = data.values() if isinstance(data, dict) else []
        out = {}
        for row in rows:
            t = str(row.get("ticker", "")).upper()
            cik = str(row.get("cik_str", "")).zfill(10)
            if t:
                out[t] = {"cik": cik, "title": row.get("title", "")}
        return out
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
    addresses = sub.get("addresses", {}) if isinstance(sub, dict) else {}
    for key in ("business", "mailing"):
        addr = addresses.get(key, {}) if isinstance(addresses, dict) else {}
        country = addr.get("country") or country
        if country:
            break
    if not country:
        country = sub.get("stateOfIncorporation") if isinstance(sub, dict) else None
        if country and len(str(country)) <= 3:
            country = "United States"

    shares = np.nan
    facts_obj = facts.get("facts", {}) if isinstance(facts, dict) else {}
    for namespace, tag in [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ]:
        unit = facts_obj.get(namespace, {}).get(tag, {}).get("units", {})
        for unit_name in ("shares", "USD"):
            arr = unit.get(unit_name, [])
            if arr:
                candidates = []
                for row in arr:
                    val = clean_num(row.get("val"))
                    if np.isfinite(val):
                        filed = pd.to_datetime(row.get("filed"), errors="coerce")
                        candidates.append((filed, val))
                if candidates:
                    candidates.sort(key=lambda z: (pd.isna(z[0]), z[0]))
                    shares = candidates[-1][1]
                    break
        if np.isfinite(shares):
            break

    return {"country": country, "shares": shares, "cik": cik, "title": item.get("title")}


@st.cache_data(ttl=900)
def get_yahoo_profile(ticker):
    try:
        payload = yahoo_get(ticker, {"range": "5d", "interval": "1d"})
        result = payload.get("chart", {}).get("result")
        meta = result[0].get("meta", {}) if result else {}
    except Exception:
        meta = {}
    return meta


def get_profile(ticker):
    sec = get_sec_profile(ticker)
    yh = get_yahoo_profile(ticker)

    shares = sec.get("shares", np.nan)
    if not np.isfinite(shares):
        shares = clean_num(yh.get("sharesOutstanding"))
    country = sec.get("country")
    if not country:
        country = yh.get("exchangeTimezoneName") or "N/A"

    return {
        "shares": shares,
        "country": country,
        "title": sec.get("title") or ticker.upper(),
    }


def parse_ibd_value(s):
    if s is None:
        return np.nan
    s = str(s).strip().upper().replace(",", "")
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*([KMBT]?)", s)
    if not m:
        return np.nan
    val = float(m.group(1))
    mult = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[m.group(2)]
    return val * mult


@st.cache_data(ttl=900)
def get_iborrowdesk(ticker):
    """Read the 1M borrow history.

    V5 prefers IBorrowDesk's JSON endpoint because the public report page can
    be rendered differently on hosted/cloud environments. It falls back to
    the report page table/text when the endpoint is unavailable.
    """
    ticker = ticker.upper().strip()
    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "application/json,text/plain,text/html,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": IBD_URL.format(ticker=ticker),
    }

    # Preferred source: IBorrowDesk's public JSON feed.
    try:
        api_url = IBD_API_URL.format(ticker=ticker)
        r = requests.get(api_url, headers=browser_headers, timeout=20)
        r.raise_for_status()
        payload = r.json()
        rows = payload.get("daily") if isinstance(payload, dict) else None
        if isinstance(rows, list) and rows:
            out = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                dt = pd.to_datetime(row.get("date"), errors="coerce")
                if pd.isna(dt):
                    continue
                fee = clean_num(row.get("fee"))
                available = parse_ibd_value(row.get("available"))
                if not np.isfinite(fee) and np.isfinite(clean_num(row.get("high_fee"))):
                    fee = clean_num(row.get("high_fee"))
                out.append({"reported": dt, "fee": fee, "available": available})
            if out:
                t = pd.DataFrame(out).drop_duplicates("reported").sort_values("reported")
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=31)
                t = t[t["reported"] >= cutoff].reset_index(drop=True)
                if not t.empty:
                    return t
    except Exception:
        pass

    # Fallback: public report page.
    try:
        url = IBD_URL.format(ticker=ticker)
        r = requests.get(url, headers=browser_headers, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception:
        return pd.DataFrame(columns=["reported", "fee", "available"])

    tables = []
    try:
        tables = pd.read_html(html)
    except Exception:
        pass

    for table in tables:
        cols = [str(c).strip().lower() for c in table.columns]
        if "reported" in cols and ("fee" in cols or "borrow fee" in cols) and any("available" in c for c in cols):
            t = table.copy()
            rename = {}
            for c in t.columns:
                lc = str(c).strip().lower()
                if lc == "reported":
                    rename[c] = "reported"
                elif "fee" in lc:
                    rename[c] = "fee"
                elif "available" in lc:
                    rename[c] = "available"
            t = t.rename(columns=rename)
            t["reported"] = pd.to_datetime(t["reported"], errors="coerce")
            t["fee"] = t["fee"].map(clean_num)
            t["available"] = t["available"].map(parse_ibd_value)
            t = t.dropna(subset=["reported"]).sort_values("reported").reset_index(drop=True)
            if not t.empty:
                t["reported"] = t["reported"].dt.tz_localize(None)
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=31)
                t = t[t["reported"] >= cutoff].reset_index(drop=True)
                if not t.empty:
                    return t

    # Final fallback: extract visible rows from page text.
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    pattern = re.compile(
        r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}\s+(?:AM|PM))\s+"
        r"([-+]?\d+(?:\.\d+)?)\s*%\s+"
        r"([0-9.,]+(?:[KMBT])?)",
        re.I,
    )
    rows = []
    for m in pattern.finditer(text):
        dt = pd.to_datetime(m.group(1), errors="coerce")
        if pd.isna(dt):
            continue
        rows.append({"reported": dt, "fee": float(m.group(2)), "available": parse_ibd_value(m.group(3))})
    if rows:
        t = pd.DataFrame(rows).drop_duplicates("reported").sort_values("reported")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=31)
        return t[t["reported"] >= cutoff].reset_index(drop=True)

    return pd.DataFrame(columns=["reported", "fee", "available"])


def latest_ibd(df):
    if df.empty:
        return None
    return df.sort_values("reported").iloc[-1]


def fmt_shares(v):
    if v is None or not np.isfinite(v):
        return "N/A"
    return f"{v:,.0f}"


def main():
    st.title("Reverse Split Scanner")
    st.caption("English-only Streamlit app for reverse-split analysis.")

    ticker = st.text_input("Ticker", value="AZI").strip().upper()
    if not ticker:
        st.stop()

    with st.spinner("Loading market and filing data..."):
        try:
            prices, event_data = get_price_data(ticker)
        except Exception as e:
            st.error(f"Could not load Yahoo Finance data: {e}")
            st.stop()

    if prices.empty:
        st.error("No price data found for this ticker.")
        st.stop()

    splits = reverse_splits(event_data.get("splits", pd.DataFrame()))
    if splits.empty:
        st.warning("No reverse split event was found in Yahoo Finance data for this ticker.")
        split_date = None
        split_ratio = "N/A"
        split_high = np.nan
    else:
        selected = splits.iloc[-1]
        split_date = pd.Timestamp(selected["date"]).normalize()
        split_ratio = f'{int(selected["numerator"])}:{int(selected["denominator"])}'
        day = prices[prices["date"] == split_date]
        if day.empty:
            day = prices[prices["date"].between(split_date - pd.Timedelta(days=3), split_date + pd.Timedelta(days=3))].head(1)
        split_high = float(day.iloc[0]["high"]) if not day.empty else np.nan

    latest_close = float(prices.iloc[-1]["close"])
    prices["rsi14"] = rsi(prices["close"], 14)
    latest_rsi = float(prices.iloc[-1]["rsi14"]) if pd.notna(prices.iloc[-1]["rsi14"]) else np.nan

    drop = np.nan
    if np.isfinite(split_high) and split_high != 0:
        drop = (latest_close / split_high - 1) * 100

    profile = get_profile(ticker)

    try:
        ibd = get_iborrowdesk(ticker)
        ibd_latest = latest_ibd(ibd)
    except Exception:
        ibd = pd.DataFrame()
        ibd_latest = None

    st.subheader(f"{profile['title']} — Reverse Split")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reverse Split Date", split_date.strftime("%Y-%m-%d") if split_date is not None else "N/A")
    c2.metric("Reverse Split Ratio", split_ratio)
    c3.metric("High on Split Day", f"${split_high:,.2f}" if np.isfinite(split_high) else "N/A")
    c4.metric("Current / Latest Close", f"${latest_close:,.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Drop from Split-Day High", f"{drop:.1f}%" if np.isfinite(drop) else "N/A")
    c6.metric("Daily RSI (14)", f"{latest_rsi:.1f}" if np.isfinite(latest_rsi) else "N/A")
    c7.metric("Shares Outstanding", fmt_shares(profile["shares"]))
    c8.metric("Company Country", profile["country"] or "N/A")

    st.divider()
    st.subheader("IBorrowDesk — 1M")

    if ibd_latest is None:
        st.info("No IBorrowDesk 1M data could be read for this ticker.")
    else:
        b1, b2, b3 = st.columns(3)
        b1.metric("Latest Borrow Fee", f"{ibd_latest['fee']:.2f}%" if pd.notna(ibd_latest["fee"]) else "N/A")
        b2.metric("Latest Shares Available", fmt_shares(ibd_latest["available"]))
        b3.metric("Last Reported", pd.Timestamp(ibd_latest["reported"]).strftime("%Y-%m-%d %H:%M"))
        chart = ibd.copy()
        if not chart.empty:
            chart = chart.set_index("reported")[["fee", "available"]]
            st.line_chart(chart, height=260)

    st.caption("IBorrowDesk 1M data uses its public JSON feed when available, with the public report page as a fallback. Values reflect IBorrowDesk data and can vary by ticker and time.")

    st.divider()
    st.subheader("Daily Price")
    chart_df = prices.set_index("date")[["close"]].rename(columns={"close": "Close"})
    st.line_chart(chart_df, height=360)

    with st.expander("Data sources and notes"):
        st.write(
            "Price history and split events: Yahoo Finance chart data. "
            "Company filing data: SEC EDGAR when available, with Yahoo fallback. "
            "Borrow data: IBorrowDesk public report page."
        )
        st.write("This app displays market data for research/education and is not investment advice.")


if __name__ == "__main__":
    main()
