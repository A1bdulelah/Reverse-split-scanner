# Reverse Split Scanner

A compact English Streamlit app for reverse-split stock research.

## Included
- Reverse split date, ratio, split-day high, latest close, and drop from split-day high
- Daily RSI(14)
- Last 30 trading days as a table only (no charts)
- IBorrowDesk full available 1-month readings: reported time, borrow fee, shares available
- Shares outstanding, previous reported shares, reported share change, free float and free-float percentage when available
- Company country
- Important recent news/events from SEC filings and Yahoo Finance search, plus Nasdaq non-compliance status when the public page exposes a ticker match
- Upcoming documented events such as an earnings date when available
- Simple positive / negative / neutral event labels; no price prediction or buy/sell recommendation
- Favorites and watchlists, including Negative Stocks
- Automatic combined lists such as Short Drop + Support and Price Drop + RSI Drop
- Watchlists persist across page refreshes by storing them in the browser URL (no database or API key required)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Community Cloud does not guarantee persistence of local server files, so this version does not rely on local SQLite files for watchlists.
- Browser URL persistence is device/browser-specific. It is intended for a simple personal watchlist. A shared cloud database can be added later if cross-device sync is needed.
- Public data can be delayed, incomplete, or unavailable for some tickers.
