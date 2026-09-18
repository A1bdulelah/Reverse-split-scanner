# Reverse Split Scanner

نسخة أولى من مشروع ويب لتحليل الأسهم بعد الـ Reverse Split.

## الموجود حالياً
- إدخال Ticker
- اكتشاف Reverse Splits
- تاريخ ونسبة آخر Reverse Split
- OHLCV ليوم التقسيم
- أعلى سعر في يوم التقسيم
- نسبة النزول من أعلى سعر إلى آخر إغلاق
- RSI يومي 14
- قائمة بالتقسيمات السابقة
- رسم السعر و RSI

## التشغيل
```bash
pip install -r requirements.txt
streamlit run app.py
```

الخطوة التالية: إضافة IBorrowDesk بإطار 1M، Shares Outstanding، وبلد الشركة.
