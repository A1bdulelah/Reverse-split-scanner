# Reverse Split Scanner v2

نسخة ويب محسنة لمشروع تحليل الأسهم بعد Reverse Split.

## الموجود
- Reverse Split date + ratio
- OHLCV يوم التقسيم
- High يوم التقسيم
- نسبة النزول من High يوم التقسيم إلى آخر إغلاق
- RSI يومي 14
- IBorrowDesk فريم 1M = آخر 30 يومًا
- آخر Borrow Fee
- آخر Shares Available
- أعلى Borrow Fee خلال 1M
- أقل Shares Available خلال 1M
- Shares Outstanding من SEC
- بلد الشركة من بيانات SEC
- رسوم السعر و RSI و Borrow Fee و Shares Available

## النشر
ضع `app.py` و`requirements.txt` في جذر GitHub repository ثم انتظر Streamlit Community Cloud ليعيد تشغيل التطبيق.

## ملاحظة
بيانات IBorrowDesk مصدر خارجي وقد يتغير API أو التغطية بمرور الوقت. التطبيق يعرض خطأ واضحًا إذا لم تتوفر البيانات بدل اختلاق قيمة.
