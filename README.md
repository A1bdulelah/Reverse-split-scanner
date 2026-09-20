# Reverse Split Scanner

A Streamlit research tool for screening a user-supplied list of stocks for reverse splits and related public-market data.

## V2

V2 is built around a multi-ticker scanner rather than a single-ticker page.

### Scanner
- Enter up to 50 tickers (one per line, comma-separated, or mixed).
- Designed for quick scans of about 10 tickers.
- Filters:
  - reverse split present
  - price range
  - days since reverse split
  - drop from split-day high
  - RSI(14)
  - reported share-count change
  - short-availability direction
  - SEC event category
- Results show ticker, price, split, drop, RSI, shares change, short direction and SEC activity.
- A matched ticker can be opened in the full Stock Detail view.

### Reliability
- HTTP timeouts and retry/backoff are used for public API requests.
- Streamlit data caching reduces repeated requests and rate-limit pressure. See the official Streamlit caching guidance: https://docs.streamlit.io/develop/concepts/architecture/caching
- Source health is visible per ticker.
- UNKNOWN is used when a source cannot establish a fact; it is not treated as a factual NO.
- Yahoo and SEC are treated as separate sources.
- SEC share-change calculations preserve the measurement date and filing date instead of comparing filing dates alone.
- IBorrowDesk is best-effort because its public page/API can change.
- Nasdaq non-compliance is not inferred from a failed scrape.

### Data sources
- Yahoo Finance public chart/search/quoteSummary endpoints for market data and news discovery.
- SEC EDGAR public submissions/XBRL endpoints for filings and share facts.
- IBorrowDesk public page for borrow-fee/availability history when readable.
- Nasdaq public non-compliant company page when its content is exposed to the app.

### Important limitations
This is a research/education tool, not a trading signal. Public feeds can be delayed, incomplete, rate-limited, or changed by their providers. A missing value should be read together with the source-health status.

The scanner does not predict price direction or provide buy/sell recommendations.
