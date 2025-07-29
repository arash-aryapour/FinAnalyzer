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

## ✨ ویژگی‌ها

### 🧠 هوش مصنوعی و یادگیری ماشین
- [x] **دسته‌بندی خودکار تراکنش‌ها** با استفاده از مدل‌های NLP پیشرفته
- [x] **تشخیص الگوهای مالی** (تراکنش‌های تکراری، اشتراک‌ها، هزینه‌های غیرمعمول)
- [x] **پیش‌بینی روندهای مالی** با مدل‌های سری زمانی
- [x] **تشخیص تقلب و فعالیت‌های مشکوک**

### 📊 تحلیل‌های پیشرفته
- [x] **داشبورد تعاملی** با نمودارهای پیشرفته
- [x] **تحلیل چندبعدی** (زمان، دسته، مبلغ، الگو)
- [x] **محاسبه شاخص‌های مالی** (نرخ پس‌انداز، نسبت‌های مالی)
- [x] **تحلیل بودجه** و پیشنهادهای بهینه‌سازی

### 🔒 امنیت و حریم خصوصی
- [x] **پردازش محلی** داده‌ها بدون ارسال به سرور
- [x] **رمزنگاری AES-256** برای داده‌های حساس
- [x] **احراز هویت دو مرحله‌ای**
- [x] **پشتیبان‌گیری و بازیابی امن**

### 🌐 یکپارچه‌سازی‌ها
- [x] **اتصال به بانک‌ها** (API رسمی بانک‌های ایرانی)
- [x] **اتصال به صرافی‌های ارز دیجیتال** (بایننس، نوبیتکس، کوکوین)
- [x] **اتصال به درگاه‌های پرداخت** (زرین‌پال، پی‌پال، استریپ)
- [x] **پشتیبانی از چندین فرمت داده** (CSV, Excel, JSON, PDF, API)

### 📱 تجربه کاربری
- [x] **رابط کاربری مدرن** و واکنش‌گرا
- [x] **پشتیبانی از چند زبان** (فارسی، انگلیسی، عربی، ترکی)
- [x] **گزارش‌گیری حرفه‌ای** (PDF, Excel, HTML)
- [x] **سیستم هشدار و اعلان** (ایمیل، تلگرام، وب‌هوک)

---

## 🎬 ساختار پروژه


```
FinAnalyzer/
├── app.py                 # اپلیکیشن اصلی Streamlit
├── core/
│   ├── __init__.py
│   ├── data_processor.py  # پردازش داده‌ها
│   ├── ml_engine.py       # موتور یادگیری ماشین
│   └── report_generator.py # تولید گزارش PDF
├── utils/
│   ├── __init__.py
│   ├── currency.py        # تبدیل ارز
│   └── notifications.py   # سیستم اعلان‌ها
├── data/
│   └── sample_data.csv    # داده‌های نمونه
├── requirements.txt       # وابستگی‌ها
└── README.md             # مستندات
```


## 🚀 نصب و راه‌اندازی

### پیش‌نیازها
- Python 3.8+
- pip
- Docker (اختیاری)

### نصب با pip
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

### نصب با Docker
```bash
# ساخت ایمیج داکر
docker build -t finanalyzer .

# اجرای کانتینر
docker run -p 8501:8501 finanalyzer
```

### اجرای نسخه دمو
```bash
# اجرای نسخه دمو با داده‌های نمونه
streamlit run app.py -- --demo
```

---

## 📖 استفاده

### 1. بارگذاری داده‌ها
- آپلود فایل‌های مالی (CSV, Excel, JSON)
- اتصال مستقیم به بانک یا صرافی
- استفاده از API برای دریافت داده‌ها

### 2. تحلیل و مشاهده نتایج
- مشاهده داشبورد تحلیل مالی
- بررسی نمودارها و بینش‌ها
- مشاهده لیست تراکنش‌ها با فیلترهای پیشرفته

### 3. تنظیم هشدارها
- تعیین محدودیت‌های بودجه
- تنظیم آستانه‌های هشدار
- انتخاب روش‌های اطلاع‌رسانی

### 4. تولید گزارش‌ها
- انتخاب نوع گزارش (مالیاتی، مدیریتی، شخصی)
- شخصی‌سازی گزارش (لوگو، رنگ‌ها)
- دانلود گزارش در فرمت مورد نظر

---

## 🛠 تکنولوژی‌های استفاده شده

### هسته اصلی
- **Python 3.8+** - زبان برنامه‌نویسی اصلی
- **Pandas & NumPy** - پردازش داده‌ها
- **Scikit-learn & Statsmodels** - یادگیری ماشین و آمار
- **Transformers (Hugging Face)** - پردازش زبان طبیعی
- **CurrencyConverter** - تبدیل ارز

### رابط کاربری
- **Streamlit** - ساخت رابط کاربری وب
- **Plotly & D3.js** - نمودارهای تعاملی
- **HTML/CSS/JavaScript** - سفارشی‌سازی UI
- **Bootstrap** - طراحی واکنش‌گرا

### زیرساخت
- **Docker** - بسته‌بندی و استقرار
- **SQLite/PostgreSQL** - پایگاه داده
- **Redis** - کش داده‌ها
- **GitHub Actions** - CI/CD

### امنیت
- **Cryptography** - رمزنگاری داده‌ها
- **PyJWT** - احراز هویت
- **OAuth2** - اتصال به سرویس‌های خارجی

---

## 🗺️ نقشه راه

### نسخه 1.0 (منتشر شده)
- [x] هسته تحلیل مالی
- [x] دسته‌بندی خودکار تراکنش‌ها
- [x] داشبورد تعاملی
- [x] گزارش‌گیری PDF
- [x] پشتیبانی از فرمت‌های رایج

---


## 📄 مجوز

این پروژه تحت مجوز MIT منتشر شده است - برای اطلاعات بیشتر فایل [LICENSE](LICENSE) را مشاهده کنید.

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
