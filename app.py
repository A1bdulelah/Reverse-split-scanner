import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Reverse Split Scanner", page_icon="📈", layout="wide")

st.title("📈 Reverse Split Scanner")
st.caption("نسخة تطويرية — بيانات تحليلية وليست توصية شراء أو بيع.")

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
IBD_URL = "https://www.iborrowdesk.com/api/ticker/{ticker}"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

HEADERS = {
    "User-Agent": "ReverseSplitScanner/1.0",
    "Accept": "application/json",
}

@st.cache_data(ttl=900)
def get_history(ticker):
    url = YAHOO_URL.format(ticker=ticker)
    params = {
        "period1": 0,
        "period2": int(datetime.now().timestamp()),
        "interval": "1d",
        "events": "history,splits",
        "includeAdjustedClose": "true",
    }
    r = requests.get(url, params=params, timeout=20,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]

    timestamps = result.get("timestamp", [])
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
        "open": q.get("open", []),
        "high": q.get("high", []),
        "low": q.get("low", []),
        "close": q.get("close", []),
        "volume": q.get("volume", []),
    }).dropna(subset=["close"])

    splits = result.get("events", {}).get("splits", {})
    split_rows = []
    for event in splits.values():
        dt = pd.to_datetime(int(event["date"]), unit="s", utc=True).tz_convert(None).normalize()
        numerator = float(event.get("numerator", 1))
        denominator = float(event.get("denominator", 1))
        split_rows.append({
            "date": dt,
            "numerator": numerator,
            "denominator": denominator,
            "ratio": f"{int(numerator) if numerator.is_integer() else numerator:g}:{int(denominator) if denominator.is_integer() else denominator:g}",
            "factor": numerator / denominator,
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
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return rsi

@st.cache_data(ttl=3600)
def get_sec_ticker_map():
    r = requests.get(SEC_TICKERS_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = []
    for item in data.values():
        rows.append({
            "ticker": str(item.get("ticker", "")).upper(),
            "cik": str(item.get("cik_str", "")).zfill(10),
            "name": item.get("title", ""),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def get_sec_company_data(ticker):
    try:
        tickers = get_sec_ticker_map()
        match = tickers[tickers["ticker"] == ticker.upper()]
        if match.empty:
            return {"name": None, "shares": None, "country": None, "cik": None}

        cik = match.iloc[0]["cik"]
        name = match.iloc[0]["name"]

        sub = requests.get(
            SEC_SUBMISSIONS_URL.format(cik=cik),
            headers=HEADERS, timeout=20
        )
        sub.raise_for_status()
        sub_data = sub.json()

        addresses = sub_data.get("addresses", {})
        business = addresses.get("business", {}) or {}
        country = business.get("country")

        facts_r = requests.get(
            SEC_FACTS_URL.format(cik=cik),
            headers=HEADERS, timeout=20
        )
        facts_r.raise_for_status()
        facts = facts_r.json()

        shares = None
        dei = facts.get("facts", {}).get("dei", {})
        concept = dei.get("EntityCommonStockSharesOutstanding", {})
        units = concept.get("units", {})
        candidates = []
        for unit_name, entries in units.items():
            for x in entries:
                if x.get("val") is not None:
                    candidates.append(x)

        if candidates:
            # Prefer the most recently reported/end-dated fact.
            candidates.sort(
                key=lambda x: (x.get("end", ""), x.get("filed", "")),
                reverse=True
            )
            shares = candidates[0].get("val")

        return {
            "name": name,
            "shares": shares,
            "country": country,
            "cik": cik,
        }
    except Exception:
        return {"name": None, "shares": None, "country": None, "cik": None}

@st.cache_data(ttl=1800)
def get_borrow_data(ticker):
    """
    IBorrowDesk daily history.
    We use the most recent 30 calendar days as the app's 1M window.
    """
    url = IBD_URL.format(ticker=ticker.upper())
    r = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20
    )
    r.raise_for_status()
    data = r.json()

    rows = data.get("daily", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["fee", "available", "high_fee", "low_fee", "high_available", "low_available"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date")
    cutoff = df["date"].max() - pd.Timedelta(days=30)
    return df[df["date"] >= cutoff].copy()

def fmt_shares(x):
    if x is None or pd.isna(x):
        return "N/A"
    x = float(x)
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:,.0f}"

ticker = st.text_input("أدخل رمز السهم", value="GNPX").strip().upper()

if st.button("تحليل السهم", type="primary"):
    if not ticker:
        st.error("أدخل رمز السهم أولاً.")
        st.stop()

    try:
        df, splits = get_history(ticker)
        if df.empty:
            st.error("لم يتم العثور على بيانات يومية.")
            st.stop()

        df["rsi14"] = rsi_wilder(df["close"], 14)
        latest = df.iloc[-1]

        # Company data
        sec = get_sec_company_data(ticker)

        st.subheader(f"📌 {ticker}")
        top1, top2, top3, top4 = st.columns(4)
        top1.metric("آخر إغلاق", f"${latest['close']:.4f}")
        top2.metric(
            "RSI اليومي (14)",
            f"{latest['rsi14']:.2f}" if pd.notna(latest["rsi14"]) else "N/A"
        )
        top3.metric("Shares Outstanding", fmt_shares(sec["shares"]))
        top4.metric("بلد الشركة", sec["country"] or "N/A")

        if sec["name"]:
            st.caption(f"اسم الشركة: {sec['name']}")

        st.divider()

        # Latest reverse split
        reverse = splits[splits["factor"] < 1].copy() if not splits.empty else pd.DataFrame()

        if not reverse.empty:
            reverse = reverse.sort_values("date")
            latest_split = reverse.iloc[-1]
            split_date = latest_split["date"]

            split_day = df[df["date"] == split_date]
            if split_day.empty:
                nearby = df[
                    (df["date"] >= split_date - pd.Timedelta(days=2)) &
                    (df["date"] <= split_date + pd.Timedelta(days=2))
                ]
                split_day = nearby.iloc[[0]] if not nearby.empty else nearby

            if not split_day.empty:
                row = split_day.iloc[-1]
                split_high = float(row["high"])
                current = float(latest["close"])
                drop_pct = ((current - split_high) / split_high) * 100

                st.subheader("🔄 آخر Reverse Split")
                a, b, c, d = st.columns(4)
                a.metric("تاريخ التقسيم", split_date.strftime("%Y-%m-%d"))
                b.metric("النسبة", latest_split["ratio"])
                c.metric("أعلى سعر يوم التقسيم", f"${split_high:.4f}")
                d.metric("النزول من الأعلى", f"{drop_pct:.2f}%")

                st.caption(
                    "النزول محسوب من أعلى سعر في يوم الـ Reverse Split إلى آخر إغلاق متاح."
                )

                st.dataframe(
                    pd.DataFrame([{
                        "التاريخ": split_date.strftime("%Y-%m-%d"),
                        "Open": row["open"],
                        "High": row["high"],
                        "Low": row["low"],
                        "Close": row["close"],
                        "Volume": row["volume"],
                    }]),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("لم يتم العثور على Reverse Split في بيانات المصدر.")

        # Borrow data
        st.divider()
        st.subheader("🩳 Short / Borrow — فريم 1M")

        try:
            borrow = get_borrow_data(ticker)
            if borrow.empty:
                st.warning("لا توجد بيانات IBorrowDesk متاحة لهذا السهم.")
            else:
                latest_b = borrow.iloc[-1]
                fee_latest = latest_b.get("fee")
                avail_latest = latest_b.get("available")

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Borrow Fee — آخر قراءة", f"{fee_latest:.2f}%" if pd.notna(fee_latest) else "N/A")
                b2.metric("Shares Available — آخر قراءة", fmt_shares(avail_latest))

                max_fee = borrow["fee"].max() if "fee" in borrow else np.nan
                min_avail = borrow["available"].min() if "available" in borrow else np.nan
                b3.metric("أعلى Fee خلال 1M", f"{max_fee:.2f}%" if pd.notna(max_fee) else "N/A")
                b4.metric("أقل Available خلال 1M", fmt_shares(min_avail))

                chart_cols = [c for c in ["fee", "available"] if c in borrow.columns]
                if "fee" in chart_cols:
                    st.caption("Borrow Fee خلال آخر 30 يومًا")
                    fee_chart = borrow.set_index("date")[["fee"]]
                    st.line_chart(fee_chart)

                if "available" in chart_cols:
                    st.caption("Shares Available خلال آخر 30 يومًا")
                    avail_chart = borrow.set_index("date")[["available"]]
                    st.line_chart(avail_chart)

                show = borrow[["date", "fee", "available"]].copy()
                show["date"] = show["date"].dt.strftime("%Y-%m-%d")
                st.dataframe(show.sort_values("date", ascending=False),
                             use_container_width=True, hide_index=True)

                st.caption(
                    "فريم 1M هنا يعني آخر 30 يومًا من تاريخ أحدث قراءة متاحة. "
                    "بيانات Borrow تعكس بيانات الإقراض المعروضة بواسطة IBorrowDesk."
                )
        except Exception as e:
            st.warning(f"تعذر جلب بيانات IBorrowDesk حاليًا: {e}")

        # Split history
        if not reverse.empty:
            st.subheader("📚 Reverse Splits السابقة")
            history_view = reverse[["date", "ratio"]].copy()
            history_view["date"] = history_view["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(history_view, use_container_width=True, hide_index=True)

        st.subheader("📈 السعر و RSI")
        chart_df = df.tail(180).set_index("date")[["close", "rsi14"]]
        st.line_chart(chart_df)

        st.caption(
            "مصادر هذه النسخة: بيانات السوق اليومية، SEC لبيانات الشركة، "
            "وIBorrowDesk لبيانات الاقتراض. قد تتغير التغطية أو حدود الوصول حسب المصدر."
        )

    except requests.HTTPError as e:
        st.error(f"تعذر جلب البيانات من أحد المصادر. HTTP: {e}")
    except Exception as e:
        st.error(f"حدث خطأ: {e}")
