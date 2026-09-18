import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Reverse Split Scanner", page_icon="📈", layout="wide")

st.title("📈 Reverse Split Scanner")
st.caption("Analytical data only — not investment advice.")

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YAHOO_SUMMARY_URL = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
IBD_URL = "https://www.iborrowdesk.com/api/ticker/{ticker}"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

HEADERS = {
    "User-Agent": "ReverseSplitScanner/1.1",
    "Accept": "application/json,text/plain,*/*",
}

def fmt_price(x):
    return "N/A" if x is None or pd.isna(x) else f"${float(x):,.2f}"

def fmt_pct(x):
    return "N/A" if x is None or pd.isna(x) else f"{float(x):.1f}%"

def fmt_shares(x):
    if x is None or pd.isna(x):
        return "N/A"
    x = float(x)
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x/1_000:.2f}K"
    return f"{x:,.0f}"

@st.cache_data(ttl=900)
def get_history(ticker):
    r = requests.get(
        YAHOO_CHART_URL.format(ticker=ticker.upper()),
        params={
            "range": "2y",
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        },
        headers=HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]

    ts = result.get("timestamp", [])
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "Date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
        "Open": q.get("open", []),
        "High": q.get("high", []),
        "Low": q.get("low", []),
        "Close": q.get("close", []),
        "Volume": q.get("volume", []),
    }).dropna(subset=["Close"]).reset_index(drop=True)

    split_rows = []
    for _, ev in result.get("events", {}).get("splits", {}).items():
        num = ev.get("numerator")
        den = ev.get("denominator")
        if num is not None and den not in (None, 0):
            split_rows.append({
                "date": pd.to_datetime(ev.get("date"), unit="s"),
                "numerator": float(num),
                "denominator": float(den),
                "factor": float(num) / float(den),
            })

    return df, pd.DataFrame(split_rows)

def rsi_wilder(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0)
    return rsi

def latest_reverse_split(split_df):
    if split_df.empty:
        return None
    rs = split_df[split_df["factor"] < 1].copy()
    if rs.empty:
        return None
    return rs.sort_values("date").iloc[-1]

@st.cache_data(ttl=1800)
def get_iborrowdesk(ticker):
    r = requests.get(
        IBD_URL.format(ticker=ticker.upper()),
        headers=HEADERS,
        timeout=20,
    )
    r.raise_for_status()
    daily = r.json().get("daily", [])

    rows = []
    for item in daily:
        date = pd.to_datetime(item.get("date"), errors="coerce")
        if pd.isna(date):
            continue
        rows.append({
            "date": date,
            "fee": item.get("fee"),
            "available": item.get("available"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("date").drop_duplicates("date", keep="last")
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    return df[df["date"] >= cutoff].reset_index(drop=True)

@st.cache_data(ttl=3600)
def get_sec_ticker_map():
    r = requests.get(SEC_TICKERS_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    return pd.DataFrame([
        {
            "ticker": str(item.get("ticker", "")).upper(),
            "cik": str(item.get("cik_str", "")).zfill(10),
            "name": item.get("title", ""),
        }
        for item in data.values()
    ])

def pick_latest_share_value(facts):
    candidates = []
    fact_root = facts.get("facts", {})

    for taxonomy, tag in [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ]:
        concept = fact_root.get(taxonomy, {}).get(tag, {})
        for unit_entries in concept.get("units", {}).values():
            for x in unit_entries:
                val = x.get("val")
                if isinstance(val, (int, float)) and val > 0:
                    candidates.append(x)

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (x.get("end", ""), x.get("filed", "")),
        reverse=True
    )
    return candidates[0]["val"]

def normalize_country(sub_data):
    addresses = sub_data.get("addresses", {}) or {}
    business = addresses.get("business", {}) or {}
    mailing = addresses.get("mailing", {}) or {}

    country = business.get("country") or mailing.get("country")
    if country:
        return country

    us_states = {
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
        "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
        "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
        "VA","WA","WV","WI","WY","DC"
    }
    state = str(sub_data.get("stateOfIncorporation", "")).upper()
    return "United States" if state in us_states else None

@st.cache_data(ttl=3600)
def get_company_profile(ticker):
    result = {
        "name": None,
        "shares": None,
        "country": None,
        "cik": None,
        "error": None,
    }

    # Yahoo fallback/first pass
    try:
        r = requests.get(
            YAHOO_SUMMARY_URL.format(ticker=ticker.upper()),
            params={"modules": "assetProfile,defaultKeyStatistics,summaryProfile"},
            headers=HEADERS,
            timeout=20,
        )
        if r.ok:
            result_list = r.json().get("quoteSummary", {}).get("result") or []
            if result_list:
                q = result_list[0]
                profile = q.get("assetProfile") or q.get("summaryProfile") or {}
                stats = q.get("defaultKeyStatistics") or {}
                shares_obj = stats.get("sharesOutstanding")
                shares = shares_obj.get("raw") if isinstance(shares_obj, dict) else None

                result["name"] = profile.get("longName") or profile.get("shortName")
                result["country"] = profile.get("country")
                result["shares"] = shares
    except Exception as e:
        result["error"] = str(e)

    # SEC fallback for missing fields
    try:
        tickers = get_sec_ticker_map()
        match = tickers[tickers["ticker"] == ticker.upper()]

        if not match.empty:
            cik = match.iloc[0]["cik"]
            result["cik"] = cik

            if not result["name"]:
                result["name"] = match.iloc[0]["name"]

            sub_r = requests.get(
                SEC_SUBMISSIONS_URL.format(cik=cik),
                headers=HEADERS,
                timeout=20,
            )
            sub_r.raise_for_status()
            sub_data = sub_r.json()

            if not result["country"]:
                result["country"] = normalize_country(sub_data)

            if result["shares"] is None:
                facts_r = requests.get(
                    SEC_FACTS_URL.format(cik=cik),
                    headers=HEADERS,
                    timeout=20,
                )
                facts_r.raise_for_status()
                result["shares"] = pick_latest_share_value(facts_r.json())
    except Exception as e:
        if not result["error"]:
            result["error"] = str(e)

    return result

def main():
    ticker = st.text_input("Ticker", value="GNPX").strip().upper()

    if not ticker:
        st.info("Enter a stock ticker.")
        return

    with st.spinner("Loading data..."):
        try:
            hist, splits = get_history(ticker)
        except Exception as e:
            st.error(f"Unable to load price data: {e}")
            return

        split = latest_reverse_split(splits)
        if split is None:
            st.warning("No reverse split was found in the available data.")
            return

        split_date = pd.Timestamp(split["date"]).normalize()
        split_ratio = f"{int(split['numerator'])}:{int(split['denominator'])}"

        split_day = hist[hist["Date"].dt.normalize() == split_date]
        if split_day.empty:
            split_day = hist[hist["Date"].dt.normalize() >= split_date].head(1)

        split_high = float(split_day["High"].iloc[0]) if not split_day.empty else None

        hist["RSI14"] = rsi_wilder(hist["Close"], 14)
        latest_close = float(hist["Close"].iloc[-1])
        latest_rsi = (
            float(hist["RSI14"].iloc[-1])
            if pd.notna(hist["RSI14"].iloc[-1])
            else None
        )
        drop_pct = (
            (latest_close - split_high) / split_high * 100
            if split_high else None
        )

        sec = get_company_profile(ticker)

        try:
            ibd = get_iborrowdesk(ticker)
        except Exception:
            ibd = pd.DataFrame()

    st.subheader(f"{ticker} — Reverse Split")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reverse Split Date", split_date.strftime("%Y-%m-%d"))
    c2.metric("Reverse Split Ratio", split_ratio)
    c3.metric("High on Split Day", fmt_price(split_high))
    c4.metric("Current / Latest Close", fmt_price(latest_close))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Drop from Split-Day High", fmt_pct(drop_pct))
    c6.metric("Daily RSI (14)", "N/A" if latest_rsi is None else f"{latest_rsi:.1f}")
    c7.metric("Shares Outstanding", fmt_shares(sec["shares"]))
    c8.metric("Company Country", sec["country"] or "N/A")

    st.divider()
    st.subheader("IBorrowDesk — 1M")

    if ibd.empty:
        st.info("No IBorrowDesk data is available for this ticker.")
    else:
        latest = ibd.sort_values("date").iloc[-1]
        b1, b2 = st.columns(2)
        b1.metric(
            "Borrow Fee",
            "N/A" if pd.isna(latest["fee"]) else f"{float(latest['fee']):.2f}%"
        )
        b2.metric("Shares Available", fmt_shares(latest["available"]))

        chart = ibd.set_index("date")[["fee", "available"]].rename(
            columns={"fee": "Borrow Fee %", "available": "Shares Available"}
        )
        st.line_chart(chart)

    st.divider()
    st.subheader("Daily Price")
    st.line_chart(hist.set_index("Date")[["Close"]])

if __name__ == "__main__":
    main()
