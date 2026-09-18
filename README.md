# Reverse Split Scanner

A Streamlit web app for analyzing stocks after reverse splits.

## Features

- Latest reverse split date
- Reverse split ratio
- High on split day
- Current / latest close
- Percentage drop from split-day high
- Daily RSI (14)
- Shares outstanding
- Company country
- IBorrowDesk 1M data
- Borrow fee
- Shares available
- Daily price chart

## Data sources

- Yahoo Finance chart data for daily prices and split events
- SEC EDGAR APIs for company profile and shares outstanding
- IBorrowDesk endpoint for borrow data

SEC EDGAR provides public submissions and XBRL company facts through data.sec.gov. API access does not require an API key, but automated access should follow SEC fair-access and user-agent guidance.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Deploy this repository as a Streamlit app and select `app.py` as the main file.

## Notes

This project is for analytical and educational use only. Data availability can vary by ticker and by source.
