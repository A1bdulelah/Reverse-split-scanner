# Reverse Split Scanner

English-only Streamlit app for reverse-split analysis.

## Included data
- Reverse split date and ratio
- Split-day high
- Latest close
- Drop from split-day high
- Daily RSI (14)
- Shares outstanding
- Company country
- IBorrowDesk 1M borrow fee and shares available
- Daily price chart

## Data sources
- Yahoo Finance for price history and split events
- SEC EDGAR APIs for company submissions and XBRL facts
- IBorrowDesk for stock-borrow data

The SEC documents its public submissions and XBRL APIs at https://www.sec.gov/search-filings/edgar-application-programming-interfaces.

## Streamlit
The repository should contain:
- app.py
- requirements.txt
- README.md

If Streamlit is configured to run `main/app.py`, place these three files inside the `main` folder instead.

This project is for analytical and educational use only.
