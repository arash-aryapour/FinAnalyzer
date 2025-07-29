<div align="center">
  <img src="https://raw.githubusercontent.com/arash-aryapour/FinAnalyzer/main/assets/logo.png" alt="FinAnalyzer Logo" width="250">

  <h1>FinAnalyzer</h1>

  <p>ابزار هوشمند تحلیل مالی شخصی و کسب‌وکاری</p>

  <p>
    <a href="https://github.com/arash-aryapour/FinAnalyzer/stargazers">
      <img src="https://img.shields.io/github/stars/arash-aryapour/FinAnalyzer?style=for-the-badge&logo=github&color=yellow" alt="GitHub stars">
    </a>
    <a href="https://github.com/arash-aryapour/FinAnalyzer/network">
      <img src="https://img.shields.io/github/forks/arash-aryapour/FinAnalyzer?style=for-the-badge&logo=github&color=blue" alt="GitHub forks">
    </a>
    <a href="https://github.com/arash-aryapour/FinAnalyzer/issues">
      <img src="https://img.shields.io/github/issues/arash-aryapour/FinAnalyzer?style=for-the-badge&logo=github&color=red" alt="GitHub issues">
    </a>
    <a href="https://github.com/arash-aryapour/FinAnalyzer/blob/main/LICENSE">
      <img src="https://img.shields.io/github/license/arash-aryapour/FinAnalyzer?style=for-the-badge&logo=github&color=green" alt="License">
    </a>
  </p>

</div>

---

## 🎯 هدف پروژه

FinAnalyzer یک پلتفرم تحلیل مالی است که به کاربران امکان می‌دهد:
- تراکنش‌های مالی خود را به صورت هوشمند دسته‌بندی کنند
- تحلیل‌های دقیق از درآمد و هزینه‌ها دریافت کنند
- گزارش‌های حرفه‌ای PDF تولید کنند
- از هوش مصنوعی برای کشف الگوهای مالی استفاده کنند

## 🏗️ ساختار پروژه

```
FinAnalyzer/
├── app.py                    # اپلیکیشن اصلی Streamlit
├── core/
│   ├── __init__.py
│   ├── data_processor.py     # پردازش و تحلیل داده‌های مالی
│   ├── ml_engine.py          # موتور یادگیری ماشین
│   └── report_generator.py  # تولید گزارش PDF
├── utils/
│   ├── __init__.py
│   ├── currency.py          # تبدیل ارزهای فیات و دیجیتال
│   └── notifications.py    # سیستم اعلان‌ها
├── data/
│   └── sample_data.csv     # داده‌های نمونه
├── requirements.txt         # وابستگی‌های پروژه
└── README.md               # مستندات
```

## 🚀 قابلیت‌های اصلی

### 1. پردازش داده‌های مالی (data_processor.py)
- **پشتیبانی از فرمت‌های مختلف**: CSV, Excel, JSON, Parquet
- **استانداردسازی داده‌ها**: تبدیل نام ستون‌ها به فرمت یکپارچه
- **تبدیل ارز**: پشتیبانی از ارزهای فیات و دیجیتال
- **تشخیص الگوها**: شناسایی تراکنش‌های تکراری، اشتراک‌ها و هزینه‌های غیرمعمول
- **تحلیل‌های مالی**: محاسبه شاخص‌هایی مانند نرخ پس‌انداز، نسبت‌های مالی

### 2. دسته‌بندی هوشمند تراکنش‌ها (ml_engine.py)
- **روش‌های دسته‌بندی چندگانه**:
  - Rule-based: استفاده از الگوهای regex برای دسته‌بندی
  - Machine Learning: مدل RandomForest آموزش‌دیده
  - Zero-shot: استفاده از مدل BART برای دسته‌بندی بدون نیاز به داده‌های آموزشی
  - Hybrid: ترکیب چند روش برای بهترین نتیجه

- **دسته‌بندی‌های پشتیبانی شده**:
  - درآمد، خرید، قبوض، حمل و نقل، غذا
  - سرمایه گذاری، سلامت، آموزش، تفریح، سایر

### 3. تولید گزارش حرفه‌ای (report_generator.py)
- **فرمت‌های خروجی**: PDF, HTML, Excel
- **پشتیبانی از زبان فارسی**: استفاده از فونت‌های مناسب برای متون فارسی
- **نمودارهای تعاملی**: ایجاد نمودارهای مالی با matplotlib و seaborn
- **قالب‌های قابل شخصی‌سازی**: استفاده از Jinja2 برای قالب‌های HTML
- **تحلیل‌های پیشرفته**: شامل تحلیل فصلی، روندها و اهداف مالی

### 4. تبدیل ارز (currency.py)
- **پشتیبانی از ارزهای فیات**: USD, EUR, GBP, JPY, IRR, TRY, CNY, RUB
- **پشتیبانی از ارزهای دیجیتال**: BTC, ETH, BNB, ADA, DOT, XRP
- **کش کردن نرخ‌ها**: ذخیره موقت نرخ‌ها برای بهبود عملکرد
- **حالت آفلاین**: استفاده از نرخ‌های پیش‌فرض در صورت عدم دسترسی به اینترنت
- **محاسبه نوسانات**: تحلیل نوسانات ارزها در دوره‌های زمانی مختلف

### 5. سیستم اعلان‌ها (notifications.py)
- **کانال‌های اطلاع‌رسانی**:
  - ایمیل (با قالب‌های HTML)
  - تلگرام (از طریق ربات)
  - وب‌هوک (برای یکپارچه‌سازی با سیستم‌های دیگر)
- **مدیریت صف**: پردازش نامتقارن اعلان‌ها
- **سیستم تلاش مجدد**: ارسال مجدد در صورت شکست
- **لاگ‌گیری کامل**: ثبت تمام فعالیت‌های سیستم اعلان‌ها

## 📊 رابط کاربری (app.py)

### داشبورد اصلی
- **کارت‌های خلاصه وضعیت مالی**:
  - درآمد کل
  - هزینه کل
  - تراز خالص
  - نرخ پس‌انداز

### نمودارهای تحلیلی
- **نمودار خطی**: روند مالی ماهانه
- **نمودار دایره‌ای**: توزیع هزینه‌ها بر اساس دسته‌بندی
- **جدول تراکنش‌ها**: نمایش فیلتر شده تراکنش‌ها با قابلیت‌های:
  - فیلتر تاریخ
  - فیلتر دسته‌بندی
  - فیلتر نوع تراکنش

### قابلیت‌های تعاملی
- **آپلود فایل**: پشتیبانی از فایل‌های CSV
- **داده‌های نمونه**: استفاده از داده‌های پیش‌فرض در صورت عدم آپلود فایل
- **تولید گزارش**: ایجاد گزارش PDF با یک کلیک
- **نمایش بینش‌ها**: نمایش تحلیل‌های هوش مصنوعی به صورت JSON

## 🛠️ تکنولوژی‌های استفاده شده

### هسته اصلی
- **Python 3.8+**: زبان برنامه‌نویسی اصلی
- **Pandas**: پردازش و تحلیل داده‌ها
- **NumPy**: محاسبات عددی
- **Scikit-learn**: الگوریتم‌های یادگیری ماشین
- **Transformers**: مدل‌های پیشرفته NLP (BART)
- **PyTorch**: فریمورک یادگیری عمیق

### رابط کاربری
- **Streamlit**: ساخت رابط کاربری وب
- **Plotly**: نمودارهای تعاملی
- **Matplotlib/Seaborn**: نمودارهای آماری

### گزارش‌گیری
- **FPDF2**: تولید فایل‌های PDF
- **Jinja2**: موتور قالب‌سازی HTML
- **ReportLab**: پیشرفته‌تر برای گزارش‌های پیچیده

### ابزارهای کمکی
- **Requests**: ارتباط با APIها
- **CurrencyConverter**: تبدیل ارز
- **python-dateutil**: پردازش تاریخ‌ها
- **Threading**: پردازش نامتقارن

## 📋 وابستگی‌ها

```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
plotly==5.17.0
scikit-learn==1.3.0
transformers==4.35.2
torch==2.1.0
fpdf2==2.7.6
requests==2.31.0
python-dateutil==2.8.2
CurrencyConverter==0.17.4
```

## 🚀 نصب و راه‌اندازی

### پیش‌نیازها
- Python 3.8+
- pip

### مراحل نصب

```bash
# کلون ریپازیتوری
git clone https://github.com/arash-aryapour/FinAnalyzer.git
cd FinAnalyzer

# ایجاد محیط مجازی
python -m venv venv
source venv/bin/activate  # در ویندوز: venv\Scripts\activate

# نصب وابستگی‌ها
pip install -r requirements.txt

# اجرای برنامه
streamlit run app.py
```

### استفاده از داده‌های نمونه

پروژه شامل یک فایل `sample_data.csv` در پوشه `data` است که می‌توانید برای تست برنامه از آن استفاده کنید.

## 📖 راهنمای استفاده

### 1. بارگذاری داده‌ها
- فایل CSV خود را آپلود کنید (با ستون‌های Date, Description, Amount, Category, Currency)
- یا از داده‌های نمونه استفاده کنید

### 2. مشاهده تحلیل‌ها
- خلاصه وضعیت مالی در کارت‌های بالا
- نمودارهای تحلیلی در بخش میانی
- جدول تراکنش‌ها با فیلترهای مختلف

### 3. تولید گزارش
- روی دکمه "تولید گزارش PDF" کلیک کنید
- گزارش را دانلود کنید

## 🔧 تنظیمات پیشرفته

### تنظیمات تبدیل ارز
```python
config = {
    'api_key': 'YOUR_API_KEY',
    'base_currency': 'USD',
    'cache_duration': 60  # دقیقه
}
```

### تنظیمات اعلان‌ها
```python
config = {
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your_email@gmail.com',
        'password': 'your_password'
    },
    'telegram': {
        'bot_token': 'YOUR_BOT_TOKEN',
        'chat_id': 'YOUR_CHAT_ID'
    }
}
```


---

## 🛠 تکنولوژی‌های استفاده شده
## 🎯 نمونه Cases of Use

### 1. تحلیل هزینه‌های شخصی
- دسته‌بندی خودکار هزینه‌های ماهانه
- شناسایی الگوهای مصرف
- پیشنهاد برای بهینه‌سازی هزینه‌ها

### 2. مدیریت بودجه کسب‌وکار
- تحلیل درآمد و هزینه‌های شرکت
- گزارش‌گیری مالیاتی
- پیش‌بینی روندهای مالی

### 3. سرمایه‌گذاری هوشمند
- ردیابی تراکنش‌های ارز دیجیتال
- تحلیل نوسانات بازار
- محاسبه سود و زیان

## 🤝 مشارکت در پروژه

ما از مشارکت‌ها استقبال می‌کنیم! لطفاً قبل از ارسال Pull Request:

1. ریپازیتوری را فورک کنید
2. یک شاخه برای ویژگی خود ایجاد کنید
3. تغییرات خود را کامیت کنید
4. شاخه را پوش کنید
5. یک Pull Request باز کنید

## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است.

---

## ❤️ حمایت از پروژه

اگر این پروژه برای شما مفید بوده، لطفاً با دادن یک ⭐️ به آن حمایت کنید!

<div align="center">

  <p>
    <a href="https://github.com/arash-aryapour">
      <img src="https://img.shields.io/badge/GitHub-arash--aryapour-black?style=for-the-badge&logo=github" alt="GitHub">
    </a>
    <a href="https://twitter.com/Arash_Ary">
      <img src="https://img.shields.io/badge/Twitter-@Arash_Ary-blue?style=for-the-badge&logo=twitter" alt="Twitter">
    </a>
    <a href="https://linkedin.com/in/arash-aryapour">
      <img src="https://img.shields.io/badge/LinkedIn-arash--aryapour-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn">
    </a>
  </p>
</div>
