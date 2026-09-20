import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Reverse Split Scanner", page_icon="🔎", layout="wide")

UA = "ReverseSplitScanner/4.0 research-app contact@example.com"
HEADERS = {"User-Agent": UA, "Accept": "application/json,text/html,*/*", "Accept-Language": "en-US,en;q=0.8"}
YCHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YSEARCH = "https://query1.finance.yahoo.com/v1/finance/search"
YSUMMARY = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUB = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
NASDAQ = "https://www.nasdaq.com/market-activity/stocks/non-compliant-company-list"
IBD = "https://www.iborrowdesk.com/report/{ticker}"

STATUS_OK = "OK"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_ERROR = "ERROR"

def num(v):
    if v is None:
        return np.nan
    try:
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        return float(str(v).replace(",", "").replace("$", "").replace("%", "").strip())
    except Exception:
        return np.nan

def human_num(v):
    if v is None:
        return np.nan
    s = str(v).strip().upper().replace(",", "")
    m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*([KMBT]?)", s)
    if not m:
        return np.nan
    return float(m.group(1)) * {"":1, "K":1e3, "M":1e6, "B":1e9, "T":1e12}[m.group(2)]

def clean_tickers(text):
    tokens = re.split(r"[\s,;]+", text.upper().strip())
    out = []
    for t in tokens:
        t = re.sub(r"[^A-Z0-9.\-]", "", t)
        if t and t not in out:
            out.append(t)
    return out[:50]

def request_json(url, params=None, timeout=12, attempts=3):
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json(), STATUS_OK, ""
        except Exception as e:
            last = str(e)
            if i < attempts - 1:
                time.sleep(0.6 * (2 ** i))
    return None, STATUS_ERROR, last or "request failed"

@st.cache_data(ttl=900, max_entries=2000)
def yahoo_chart(ticker, range_value="1y"):
    payload, status, err = request_json(YCHART.format(ticker=ticker), {"range": range_value, "interval":"1d", "events":"div,splits"})
    if status != STATUS_OK or not payload:
        return {"status": status, "error": err, "df": pd.DataFrame(), "splits": pd.DataFrame(), "meta": {}}
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        return {"status": STATUS_UNKNOWN, "error":"No chart result", "df":pd.DataFrame(), "splits":pd.DataFrame(), "meta":{}}
    ts = result.get("timestamp") or []
    q = (result.get("indicators", {}).get("quote") or [{}])[0]
    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
        "open": q.get("open", []), "high": q.get("high", []),
        "low": q.get("low", []), "close": q.get("close", []), "volume": q.get("volume", [])
    })
    if not df.empty:
        df["date"] = df["date"].dt.normalize()
        df = df.dropna(subset=["close"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    splits = []
    for x in (result.get("events", {}).get("splits", {}) or {}).values():
        if x.get("date") is not None:
            splits.append({"date":pd.to_datetime(x["date"], unit="s").normalize(),
                           "numerator":num(x.get("numerator")), "denominator":num(x.get("denominator"))})
    return {"status":STATUS_OK, "error":"", "df":df, "splits":pd.DataFrame(splits), "meta":result.get("meta", {})}

def rsi(series, period=14):
    d = series.diff()
    gain, loss = d.clip(lower=0), -d.clip(upper=0)
    ag = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - 100/(1+rs)
    return out.where(~((al==0)&(ag>0)),100).where(~((ag==0)&(al>0)),0)

def reverse_splits(split_df):
    if split_df.empty:
        return split_df
    x = split_df.copy()
    x["numerator"] = pd.to_numeric(x["numerator"], errors="coerce")
    x["denominator"] = pd.to_numeric(x["denominator"], errors="coerce")
    return x[(x["numerator"] > 0) & (x["denominator"] > 0) & (x["numerator"] < x["denominator"])].sort_values("date")

@st.cache_data(ttl=3600, max_entries=5)
def sec_ticker_map():
    data, status, _ = request_json(SEC_TICKERS, timeout=15)
    if status != STATUS_OK or not isinstance(data, dict):
        return {}
    return {str(v.get("ticker","")).upper(): {"cik":str(v.get("cik_str","")).zfill(10), "title":v.get("title","")}
            for v in data.values() if v.get("ticker")}

@st.cache_data(ttl=1800, max_entries=2000)
def sec_profile(ticker):
    item = sec_ticker_map().get(ticker)
    if not item:
        return {"status":STATUS_UNKNOWN, "error":"Ticker not found in SEC ticker map", "cik":None, "title":ticker, "shares":np.nan, "sub":{}}
    sub, ss, se = request_json(SEC_SUB.format(cik=item["cik"]), timeout=15)
    if ss != STATUS_OK:
        return {"status":ss, "error":se, "cik":item["cik"], "title":item["title"], "shares":np.nan, "sub":{}}
    facts, fs, fe = request_json(SEC_FACTS.format(cik=item["cik"]), timeout=15)
    shares = np.nan
    if fs == STATUS_OK and isinstance(facts, dict):
        units = facts.get("facts",{}).get("dei",{}).get("EntityCommonStockSharesOutstanding",{}).get("units",{})
        rows = []
        for arr in units.values():
            for row in arr:
                v = num(row.get("val"))
                if np.isfinite(v):
                    rows.append((pd.to_datetime(row.get("filed"), errors="coerce"), pd.to_datetime(row.get("end"), errors="coerce"), v))
        if rows:
            rows.sort(key=lambda x:(pd.isna(x[0]), x[0], pd.isna(x[1]), x[1]))
            shares = rows[-1][2]
    return {"status":STATUS_OK, "error":"", "cik":item["cik"], "title":item["title"], "shares":shares, "sub":sub}

@st.cache_data(ttl=1200, max_entries=2000)
def yahoo_profile(ticker):
    payload, status, err = request_json(YSUMMARY.format(ticker=ticker), {"modules":"defaultKeyStatistics,price,calendarEvents"}, timeout=12)
    if status != STATUS_OK or not payload:
        return {"status":status, "error":err, "shares":np.nan, "float":np.nan, "earnings":None}
    res = (payload.get("quoteSummary",{}).get("result") or [None])[0]
    if not res:
        return {"status":STATUS_UNKNOWN, "error":"No quote summary", "shares":np.nan, "float":np.nan, "earnings":None}
    stats = res.get("defaultKeyStatistics", {})
    def raw(k):
        v = stats.get(k)
        return v.get("raw") if isinstance(v,dict) else v
    earnings = (res.get("calendarEvents",{}).get("earnings",{}).get("earningsDate") or [])
    return {"status":STATUS_OK, "error":"", "shares":num(raw("sharesOutstanding")), "float":num(raw("floatShares")),
            "earnings": earnings[0].get("raw") if earnings and isinstance(earnings[0],dict) else (earnings[0] if earnings else None)}

def shares_change(ticker):
    p = sec_profile(ticker)
    sub = p.get("sub",{})
    recent = sub.get("filings",{}).get("recent",{}) if isinstance(sub,dict) else {}
    forms, filed, acc = recent.get("form",[]), recent.get("filingDate",[]), recent.get("accessionNumber",[])
    rows = []
    for i, form in enumerate(forms):
        if form not in {"10-K","10-K/A","10-Q","10-Q/A","8-K","8-K/A","S-1","S-1/A","S-3","S-3/A","424B3","424B5"}:
            continue
        # Share measurements come from XBRL facts; filing history is only used for display context.
        rows.append((pd.to_datetime(filed[i],errors="coerce"), form))
    facts, status, _ = request_json(SEC_FACTS.format(cik=p.get("cik")), timeout=15) if p.get("cik") else ({}, STATUS_ERROR, "")
    vals = []
    if status == STATUS_OK:
        units = facts.get("facts",{}).get("dei",{}).get("EntityCommonStockSharesOutstanding",{}).get("units",{}).get("shares",[])
        for r in units:
            v, end, fd = num(r.get("val")), pd.to_datetime(r.get("end"),errors="coerce"), pd.to_datetime(r.get("filed"),errors="coerce")
            if np.isfinite(v) and pd.notna(end):
                vals.append((end,fd,v,r.get("form","")))
    if not vals:
        return {"latest":np.nan,"previous":np.nan,"change_pct":np.nan,"measurement_date":None,"filed_date":None,"status":status}
    x = pd.DataFrame(vals,columns=["measurement","filed","value","form"]).sort_values(["measurement","filed"]).drop_duplicates("measurement",keep="last")
    latest = x.iloc[-1]
    prev = x.iloc[-2] if len(x)>1 else None
    pct = np.nan if prev is None or prev["value"]==0 else (latest["value"]/prev["value"]-1)*100
    return {"latest":latest["value"],"previous":prev["value"] if prev is not None else np.nan,
            "change_pct":pct,"measurement_date":latest["measurement"],"filed_date":latest["filed"],"status":STATUS_OK}

@st.cache_data(ttl=900, max_entries=2000)
def iborrow(ticker):
    try:
        r = requests.get(IBD.format(ticker=ticker), headers={**HEADERS,"User-Agent":"Mozilla/5.0 Chrome/128"}, timeout=12)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for t in tables:
            cols = {str(c).lower():c for c in t.columns}
            rep = next((c for k,c in cols.items() if "reported" in k),None)
            fee = next((c for k,c in cols.items() if "fee" in k),None)
            avail = next((c for k,c in cols.items() if "available" in k),None)
            if rep and fee and avail:
                out = pd.DataFrame({"reported":pd.to_datetime(t[rep],errors="coerce"),"fee":t[fee].map(num),"available":t[avail].map(human_num)})
                out = out.dropna(subset=["reported"]).sort_values("reported")
                return {"status":STATUS_OK,"error":"","df":out.tail(31)}
    except Exception as e:
        return {"status":STATUS_UNKNOWN,"error":str(e),"df":pd.DataFrame(columns=["reported","fee","available"])}
    return {"status":STATUS_UNKNOWN,"error":"No readable table","df":pd.DataFrame(columns=["reported","fee","available"])}

@st.cache_data(ttl=1800, max_entries=2000)
def news_data(ticker):
    payload,status,err = request_json(YSEARCH, {"q":ticker,"quotesCount":3,"newsCount":10,"enableFuzzyQuery":"false"}, timeout=12)
    if status != STATUS_OK or not payload:
        return {"status":status,"error":err,"df":pd.DataFrame()}
    rows=[]
    for x in payload.get("news",[]):
        dt=pd.to_datetime(x.get("providerPublishTime"),unit="s",errors="coerce")
        if pd.notna(dt):
            rows.append({"date":dt,"title":x.get("title",""),"publisher":x.get("publisher","Yahoo Finance"),"url":x.get("link","")})
    return {"status":STATUS_OK,"error":"","df":pd.DataFrame(rows).sort_values("date",ascending=False) if rows else pd.DataFrame()}

@st.cache_data(ttl=1800, max_entries=2000)
def sec_events(ticker):
    p=sec_profile(ticker); sub=p.get("sub",{})
    recent=sub.get("filings",{}).get("recent",{}) if isinstance(sub,dict) else {}
    rows=[]
    forms=recent.get("form",[]); dates=recent.get("filingDate",[]); acc=recent.get("accessionNumber",[]); docs=recent.get("primaryDocument",[])
    for i,form in enumerate(forms[:100]):
        if form not in {"8-K","8-K/A","S-1","S-1/A","S-3","S-3/A","424B3","424B5","25","25-NSE","10-Q","10-K"}: continue
        dt=pd.to_datetime(dates[i],errors="coerce")
        if pd.isna(dt) or dt < pd.Timestamp.now().normalize()-pd.Timedelta(days=60): continue
        accession=str(acc[i]).replace("-","")
        doc=docs[i] if i<len(docs) else ""
        url=f"https://www.sec.gov/Archives/edgar/data/{int(p['cik'])}/{accession}/{doc}" if doc else f"https://www.sec.gov/edgar/browse/?CIK={int(p['cik'])}"
        category = "Financing / Registration" if form.startswith(("S-","424")) else ("Listing Risk" if form.startswith("25") else "SEC Filing")
        rows.append({"date":dt,"form":form,"category":category,"url":url})
    return {"status":p.get("status",STATUS_UNKNOWN),"error":"","df":pd.DataFrame(rows)}

def source_badge(status):
    return {"OK":"🟢","UNKNOWN":"🟡","ERROR":"🔴"}.get(status,"⚪")

def analyze(ticker):
    out={"ticker":ticker,"errors":[]}
    y=yahoo_chart(ticker,"2y"); out["sources"]={"Yahoo":y["status"]}
    df=y["df"]
    if df.empty:
        out.update({"valid":False,"reason":"No Yahoo price data","price":np.nan})
        return out
    df["rsi14"]=rsi(df["close"])
    latest=df.iloc[-1]
    splits=reverse_splits(y["splits"])
    split_date=None; ratio="—"; split_high=np.nan
    if not splits.empty:
        s=splits.iloc[-1]; split_date=pd.Timestamp(s["date"]).normalize()
        ratio=f'{int(s["numerator"])}:{int(s["denominator"])}'
        day=df[df["date"]==split_date]
        if not day.empty: split_high=num(day.iloc[0]["high"])
    price=num(latest["close"]); rv=num(latest["rsi14"])
    drop=(price/split_high-1)*100 if np.isfinite(split_high) and split_high else np.nan
    sec=sec_profile(ticker); yp=yahoo_profile(ticker); sh=shares_change(ticker); ib=iborrow(ticker)
    ns=news_data(ticker); se=sec_events(ticker)
    out.update({"valid":True,"price":price,"rsi":rv,"split_date":split_date,"ratio":ratio,"split_high":split_high,
                "drop":drop,"shares":sec.get("shares") if np.isfinite(sec.get("shares",np.nan)) else yp.get("shares"),
                "float":yp.get("float"),"share_change":sh,"short":ib["df"],"news":ns["df"],"sec_events":se["df"],
                "sources":{"Yahoo":y["status"],"SEC":sec.get("status"),"Shares":sh.get("status"),"Short":ib["status"],"News":ns["status"],"SEC Events":se["status"]},
                "title":sec.get("title",ticker)})
    return out

def scan_all(tickers):
    results=[]
    with ThreadPoolExecutor(max_workers=min(5,len(tickers))) as ex:
        futures={ex.submit(analyze,t):t for t in tickers}
        for f in as_completed(futures):
            try: results.append(f.result())
            except Exception as e: results.append({"ticker":futures[f],"valid":False,"reason":str(e),"sources":{}})
    order={t:i for i,t in enumerate(tickers)}
    return sorted(results,key=lambda x:order.get(x["ticker"],999))

def filter_match(x, cfg):
    checks=[]
    if cfg["only_split"]: checks.append(pd.notna(x.get("split_date")))
    if np.isfinite(cfg["min_price"]): checks.append(np.isfinite(x.get("price",np.nan)) and x["price"]>=cfg["min_price"])
    if np.isfinite(cfg["max_price"]): checks.append(np.isfinite(x.get("price",np.nan)) and x["price"]<=cfg["max_price"])
    if cfg["days_min"] is not None or cfg["days_max"] is not None:
        d=(pd.Timestamp.now().normalize()-x["split_date"]).days if pd.notna(x.get("split_date")) else -1
        if cfg["days_min"] is not None: checks.append(d>=cfg["days_min"])
        if cfg["days_max"] is not None: checks.append(d<=cfg["days_max"])
    if cfg["drop_min"] is not None: checks.append(np.isfinite(x.get("drop",np.nan)) and x["drop"]<=cfg["drop_min"])
    if cfg["rsi_max"] is not None: checks.append(np.isfinite(x.get("rsi",np.nan)) and x["rsi"]<=cfg["rsi_max"])
    if cfg["shares_change_max"] is not None:
        v=x.get("share_change",{}).get("change_pct",np.nan); checks.append(np.isfinite(v) and v<=cfg["shares_change_max"])
    if cfg["short_mode"]!="Any":
        s=x.get("short",pd.DataFrame())
        if s.empty: checks.append(False)
        else:
            vals=s["available"].dropna()
            if len(vals)<2: checks.append(False)
            else:
                delta=vals.iloc[-1]-vals.iloc[0]
                checks.append(delta<0 if cfg["short_mode"]=="Falling availability" else delta>0)
    if cfg["sec_category"]!="Any":
        ev=x.get("sec_events",pd.DataFrame())
        checks.append(cfg["sec_category"] in set(ev.get("category",[])))
    return all(checks) if checks else True

def fmt(v, suffix=""):
    return "—" if v is None or not np.isfinite(num(v)) else f"{num(v):,.2f}{suffix}"

def main():
    st.markdown("## 🔎 Reverse Split Scanner")
    st.caption("Research scanner for reverse-split, price, RSI, shares, short-availability and SEC-event data. Filters describe data; they do not provide buy/sell recommendations.")
    tabs=st.tabs(["Scanner","Stock Detail","Data Health"])
    with tabs[0]:
        left,right=st.columns([2,1])
        with left:
            tickers_text=st.text_area("Tickers — one per line, comma-separated, or both", value="GNPX\nSMTK", height=130)
        with right:
            st.markdown("**Scanner limits**")
            st.caption("Up to 50 tickers per scan. For the fastest results, start with 10.")
            only_split=st.checkbox("Has reverse split",True)
            min_price=st.number_input("Min price",min_value=0.0,value=0.0,step=0.1)
            max_price=st.number_input("Max price",min_value=0.0,value=0.0,step=0.1)
        with st.expander("Advanced filters"):
            c1,c2,c3=st.columns(3)
            days_min=c1.number_input("Days since split — min",min_value=0,value=0,step=1)
            days_max=c1.number_input("Days since split — max (0 = off)",min_value=0,value=0,step=1)
            drop_min=c2.number_input("Drop from split-day high — max %",min_value=-1000.0,max_value=100.0,value=0.0,step=1.0)
            rsi_max=c2.number_input("RSI(14) — max (0 = off)",min_value=0.0,max_value=100.0,value=0.0,step=1.0)
            shares_max=c3.number_input("Share change — max % (0 = off)",min_value=-1000.0,max_value=1000.0,value=0.0,step=1.0)
            short_mode=c3.selectbox("Short availability",["Any","Falling availability","Rising availability"])
            sec_cat=c3.selectbox("SEC event category",["Any","Financing / Registration","Listing Risk","SEC Filing"])
        tickers=clean_tickers(tickers_text)
        cfg={"only_split":only_split,"min_price":min_price if min_price>0 else np.nan,"max_price":max_price if max_price>0 else np.nan,
             "days_min":days_min if days_min>0 else None,"days_max":days_max if days_max>0 else None,
             "drop_min":drop_min if drop_min!=0 else None,"rsi_max":rsi_max if rsi_max!=0 else None,
             "shares_change_max":shares_max if shares_max!=0 else None,"short_mode":short_mode,"sec_category":sec_cat}
        if st.button("🔍 SCAN",type="primary",use_container_width=True):
            if not tickers: st.error("Enter at least one ticker.")
            else:
                with st.spinner(f"Scanning {len(tickers)} ticker(s)…"):
                    st.session_state["scan_results"]=scan_all(tickers)
                    st.session_state["scan_cfg"]=cfg
        results=st.session_state.get("scan_results",[])
        if results:
            valid=[x for x in results if x.get("valid")]
            matched=[x for x in valid if filter_match(x,cfg)]
            st.markdown("### Scan Summary")
            a,b,c,d=st.columns(4)
            a.metric("Scanned",len(results)); b.metric("Valid market data",len(valid)); c.metric("Reverse splits",sum(pd.notna(x.get("split_date")) for x in valid)); d.metric("Matched all filters",len(matched))
            rows=[]
            for x in matched:
                rows.append({"Ticker":x["ticker"],"Price":fmt(x.get("price"),""),"Reverse Split":f'{x.get("ratio","—")} · {x["split_date"].date()}' if pd.notna(x.get("split_date")) else "—",
                             "Drop":fmt(x.get("drop"),"%"),"RSI":fmt(x.get("rsi")),"Shares Δ":fmt(x.get("share_change",{}).get("change_pct"),"%"),
                             "Short":("—" if x.get("short",pd.DataFrame()).empty else "↓" if len(x["short"])>1 and x["short"]["available"].iloc[-1]<x["short"]["available"].iloc[0] else "↑"),
                             "SEC":"⚠️" if not x.get("sec_events",pd.DataFrame()).empty else "—"})
            st.markdown(f"**{len(results)} scanned → {sum(pd.notna(x.get('split_date')) for x in valid)} had reverse splits → {len(matched)} matched all active filters**")
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,height=420)
            if matched:
                opts=[x["ticker"] for x in matched]
                chosen=st.selectbox("Open stock detail",opts)
                st.session_state["detail_ticker"]=chosen
        elif "scan_results" not in st.session_state:
            st.info("Enter your tickers, choose filters, then press SCAN.")

    with tabs[1]:
        ticker=st.text_input("Ticker for full detail",value=st.session_state.get("detail_ticker","GNPX")).strip().upper()
        if st.button("Load detail"):
            st.session_state["detail_ticker"]=ticker
        ticker=st.session_state.get("detail_ticker",ticker)
        if ticker:
            with st.spinner("Loading detail…"): x=analyze(ticker)
            if not x.get("valid"):
                st.error(x.get("reason","No data"))
            else:
                st.markdown(f"### {x.get('title',ticker)} · {ticker}")
                m=st.columns(6)
                vals=[("Price",fmt(x["price"])),("RSI",fmt(x["rsi"])),("Split",x["ratio"]),("Split date",x["split_date"].date() if pd.notna(x["split_date"]) else "—"),("Drop",fmt(x["drop"],"%")),("Shares Δ",fmt(x["share_change"].get("change_pct"),"%"))]
                for col,(lab,val) in zip(m,vals): col.metric(lab,val)
                st.markdown("#### Last 30 trading days")
                p=yahoo_chart(ticker,"2y")["df"].tail(30).sort_values("date",ascending=False).copy()
                p["Change %"]=p["close"].pct_change(periods=-1)*100
                st.dataframe(p.rename(columns={"date":"Date","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})[["Date","Open","High","Low","Close","Volume","Change %"]],hide_index=True,use_container_width=True)
                st.markdown("#### Short availability")
                s=x.get("short",pd.DataFrame())
                st.dataframe(s.sort_values("reported",ascending=False) if not s.empty else pd.DataFrame({"Status":["Unavailable from public IBorrowDesk page"]}),hide_index=True,use_container_width=True)
                st.markdown("#### SEC events / News")
                ev=x.get("sec_events",pd.DataFrame()); nw=x.get("news",pd.DataFrame())
                if not ev.empty: st.dataframe(ev,use_container_width=True,hide_index=True)
                if not nw.empty: st.dataframe(nw[["date","title","publisher","url"]],use_container_width=True,hide_index=True)
    with tabs[2]:
        results=st.session_state.get("scan_results",[])
        if not results: st.info("Run a scan first to see per-ticker source health.")
        else:
            rows=[]
            for x in results:
                src=x.get("sources",{})
                rows.append({"Ticker":x["ticker"],**{k:source_badge(v)+" "+v for k,v in src.items()}})
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            st.caption("🟢 source returned usable data · 🟡 source unavailable/ambiguous · 🔴 request failed. UNKNOWN is intentionally different from a factual NO.")

if __name__=="__main__":
    main()
