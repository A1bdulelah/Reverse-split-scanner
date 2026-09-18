import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

st.set_page_config(page_title="Reverse Split Scanner", page_icon="📈", layout="wide")

st.title("📈 Reverse Split Scanner")
st.caption("نسخة تجريبية — تحليل بيانات السهم، وليست توصية شراء أو بيع.")

API_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

@st.cache_data(ttl=900)
def get_history(ticker):
    url = API_URL.format(ticker=ticker)
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
    data = r.json()["chart"]["result"][0]

    timestamps = data.get("timestamp", [])
    q = data["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
        "open": q.get("open", []),
        "high": q.get("high", []),
        "low": q.get("low", []),
        "close": q.get("close", []),
        "volume": q.get("volume", []),
    }).dropna(subset=["close"])

    splits = data.get("events", {}).get("splits", {})
    split_rows = []
    for _, event in splits.items():
        # Yahoo uses numerator/denominator; 1:22 is represented as 1/22.
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

    # Handle all-gain / all-loss edge cases cleanly.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return rsi

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

        st.subheader(f"البيانات الحالية — {ticker}")
        c1, c2, c3 = st.columns(3)
        c1.metric("آخر إغلاق", f"${latest['close']:.4f}")
        c2.metric("RSI اليومي (14)", f"{latest['rsi14']:.2f}" if pd.notna(latest["rsi14"]) else "N/A")
        c3.metric("آخر تاريخ", latest["date"].strftime("%Y-%m-%d"))

        st.divider()

        # Reverse splits only: numerator < denominator.
        reverse = splits[splits["factor"] < 1].copy()

        if reverse.empty:
            st.info("لم يتم العثور على Reverse Split في بيانات المصدر.")
            st.stop()

        reverse = reverse.sort_values("date")
        latest_split = reverse.iloc[-1]

        split_date = latest_split["date"]
        split_day = df[df["date"] == split_date]

        if split_day.empty:
            # Sometimes timestamps/source calendars differ by one day.
            nearby = df[(df["date"] >= split_date - pd.Timedelta(days=2)) &
                        (df["date"] <= split_date + pd.Timedelta(days=2))]
            split_day = nearby.iloc[[0]] if not nearby.empty else nearby

        if split_day.empty:
            st.warning("تم العثور على Reverse Split لكن لم نجد شمعة يوم التقسيم في نفس المصدر.")
            st.stop()

        row = split_day.iloc[-1]
        split_high = float(row["high"])
        current = float(latest["close"])
        drop_pct = ((current - split_high) / split_high) * 100

        st.subheader("آخر Reverse Split")
        a, b, c, d = st.columns(4)
        a.metric("تاريخ التقسيم", split_date.strftime("%Y-%m-%d"))
        b.metric("النسبة", latest_split["ratio"])
        c.metric("أعلى سعر يوم التقسيم", f"${split_high:.4f}")
        d.metric("النزول من أعلى سعر", f"{drop_pct:.2f}%")

        st.caption("النزول محسوب من أعلى سعر في شمعة يوم الـ Reverse Split إلى آخر إغلاق متاح.")

        st.subheader("شمعة يوم التقسيم")
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

        st.subheader("Reverse Splits السابقة")
        history_view = reverse[["date", "ratio"]].copy()
        history_view["date"] = history_view["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(history_view, use_container_width=True, hide_index=True)

        st.subheader("السعر و RSI")
        chart_df = df.tail(180).set_index("date")[["close", "rsi14"]]
        st.line_chart(chart_df)

        st.info(
            "النسخة الأولى تركز على OHLC + Reverse Split + RSI. "
            "سنضيف في الخطوة التالية بيانات IBorrowDesk الشهرية، "
            "Shares Outstanding، وبلد الشركة مع مصادر واضحة."
        )

    except requests.HTTPError as e:
        st.error(f"تعذر جلب البيانات من المصدر. HTTP: {e}")
    except Exception as e:
        st.error(f"حدث خطأ: {e}")

st.divider()
st.caption("مصادر البيانات قد تتغير أو تضع حدوداً على الوصول البرمجي؛ راجع شروط الاستخدام قبل النشر العام.")
