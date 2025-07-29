# core/data_processor.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re
from currency_converter import CurrencyConverter
import warnings
warnings.filterwarnings('ignore')

class FinancialDataProcessor:
    """
    کلاس اصلی برای پردازش و تحلیل داده‌های مالی
    شامل بارگذاری، استانداردسازی، تبدیل ارز، تشخیص الگوها و تولید بینش‌های مالی
    """
    
    def __init__(self, config: Dict = None):
        """
        مقداردهی اولیه پردازشگر داده‌های مالی
        
        Args:
            config (Dict): تنظیمات سفارشی برای پردازشگر
        """
        self.config = config or {}
        self.currency_converter = CurrencyConverter()
        self.categories = [
            "درآمد", "خرید", "قبوض", "حمل و نقل", "غذا",
            "سرمایه گذاری", "سلامت", "آموزش", "تفریح", "سایر"
        ]
        
        # تنظیمات پیش‌فرض
        self.default_currency = self.config.get('default_currency', 'USD')
        self.date_format = self.config.get('date_format', '%Y-%m-%d')
        self.iqr_multiplier = self.config.get('iqr_multiplier', 1.5)
        
    def load_data(self, file_path: str, file_type: str = 'auto') -> pd.DataFrame:
        """
        بارگذاری داده‌ها از فایل‌های مختلف
        
        Args:
            file_path (str): مسیر فایل ورودی
            file_type (str): نوع فایل (csv, xlsx, json, parquet)
            
        Returns:
            pd.DataFrame: دیتافریم استاندارد شده
        """
        if file_type == 'auto':
            file_type = file_path.split('.')[-1].lower()
            
        loaders = {
            'csv': self._load_csv,
            'xlsx': self._load_excel,
            'xls': self._load_excel,
            'json': self._load_json,
            'parquet': self._load_parquet
        }
        
        if file_type not in loaders:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        df = loaders[file_type](file_path)
        return self._standardize_columns(df)
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """بارگذاری فایل CSV با پارامترهای بهینه"""
        return pd.read_csv(
            file_path,
            encoding='utf-8',
            thousands=',',
            decimal='.',
            parse_dates=False,
            infer_datetime_format=False
        )
    
    def _load_excel(self, file_path: str) -> pd.DataFrame:
        """بارگذاری فایل Excel"""
        return pd.read_excel(file_path)
    
    def _load_json(self, file_path: str) -> pd.DataFrame:
        """بارگذاری فایل JSON"""
        return pd.read_json(file_path)
    
    def _load_parquet(self, file_path: str) -> pd.DataFrame:
        """بارگذاری فایل Parquet"""
        return pd.read_parquet(file_path)
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        استانداردسازی نام ستون‌ها به فرمت یکپارچه
        
        Args:
            df (pd.DataFrame): دیتافریم ورودی
            
        Returns:
            pd.DataFrame: دیتافریم با ستون‌های استاندارد
        """
        column_mapping = {
            'date': ['Date', 'تاریخ', 'datetime', 'time', 'Transaction Date', 'تاریخ تراکنش'],
            'description': ['Description', 'توضیحات', 'desc', 'memo', 'Note', 'یادداشت'],
            'amount': ['Amount', 'مبلغ', 'value', 'price', 'Quantity', 'مقدار'],
            'category': ['Category', 'دسته‌بندی', 'type', 'tag', 'Class', 'طبقه'],
            'currency': ['Currency', 'ارز', 'curr', 'Unit', 'واحد']
        }
        
        standardized = {}
        for std_col, possible_cols in column_mapping.items():
            for col in possible_cols:
                if col in df.columns:
                    standardized[std_col] = df[col]
                    break
        
        # اگر ستونی پیدا نشد، ایجاد ستون خالی
        for std_col in column_mapping.keys():
            if std_col not in standardized:
                standardized[std_col] = None
        
        return pd.DataFrame(standardized)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        پیش‌پردازش کامل داده‌های مالی
        
        Args:
            df (pd.DataFrame): دیتافریم خام
            
        Returns:
            pd.DataFrame: دیتافریم پردازش شده
        """
        # کپی داده‌ها برای جلوگیری از تغییر داده‌های اصلی
        df = df.copy()
        
        # تبدیل تاریخ
        df = self._process_dates(df)
        
        # تبدیل ارز
        df = self._convert_currencies(df)
        
        # استخراج اطلاعات زمانی
        df = self._extract_temporal_features(df)
        
        # تشخیص نوع تراکنش
        df = self._determine_transaction_type(df)
        
        # پر کردن مقادیر خالی دسته‌بندی
        df = self._fill_missing_categories(df)
        
        return df
    
    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """پردازش ستون تاریخ"""
        if 'date' not in df.columns:
            df['date'] = pd.NaT
            return df
            
        # تلاش برای تبدیل تاریخ با فرمت‌های مختلف
        date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y',
            '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y',
            '%Y%m%d', '%d%m%Y', '%m%d%Y'
        ]
        
        for fmt in date_formats:
            try:
                df['date'] = pd.to_datetime(df['date'], format=fmt, errors='coerce')
                if df['date'].notna().any():
                    break
            except:
                continue
        
        # اگر هنوز تاریخ‌های نامعتبر وجود دارد، با روش پیش‌فرض پانداس تلاش کن
        if df['date'].isna().all():
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df
    
    def _convert_currencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """تبدیل مبالغ به ارز پایه (USD)"""
        if 'amount' not in df.columns:
            df['amount_usd'] = 0.0
            return df
            
        if 'currency' not in df.columns or df['currency'].isna().all():
            df['currency'] = self.default_currency
        
        # تبدیل مقادیر عددی
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        
        # تبدیل به USD
        df['amount_usd'] = df.apply(
            lambda x: self._convert_to_usd(x['amount'], x['currency']), 
            axis=1
        )
        
        return df
    
    def _convert_to_usd(self, amount: float, currency: str) -> float:
        """تبدیل مبلغ به دلار آمریکا"""
        try:
            if currency.upper() == 'IRR':
                # نرخ تبدیل تقریبی ریال به دلار
                return amount / 50000
            elif currency.upper() == 'USD':
                return amount
            else:
                return self.currency_converter.convert(amount, currency, 'USD')
        except:
            return amount
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """استخراج ویژگی‌های زمانی"""
        if 'date' not in df.columns:
            return df
            
        # حذف ردیف‌های بدون تاریخ معتبر
        df = df[df['date'].notna()]
        
        # استخراج اطلاعات زمانی
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.day_name()
        df['week'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # ماه شمسی برای کاربران ایرانی
        df['month_fa'] = df['date'].apply(self._get_persian_month)
        
        return df
    
    def _get_persian_month(self, date: datetime) -> str:
        """تبدیل تاریخ به ماه شمسی"""
        try:
            import jdatetime
            jd = jdatetime.date.fromgregorian(date=date)
            months = [
                'فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور',
                'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند'
            ]
            return months[jd.month - 1]
        except:
            return str(date.month)
    
    def _determine_transaction_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """تشخیص نوع تراکنش (درآمد یا هزینه)"""
        if 'amount' not in df.columns:
            df['type'] = 'سایر'
            return df
            
        df['type'] = np.where(df['amount'] > 0, 'درآمد', 'هزینه')
        return df
    
    def _fill_missing_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """پر کردن مقادیر خالی دسته‌بندی با قوانین ساده"""
        if 'category' not in df.columns:
            df['category'] = 'سایر'
            return df
            
        # پر کردن مقادیر خالی با دسته‌بندی پیش‌فرض
        df['category'] = df['category'].fillna('سایر')
        
        # اعمال قوانین ساده برای دسته‌بندی
        for idx, row in df.iterrows():
            if row['category'] == 'سایر' and pd.notna(row['description']):
                df.at[idx, 'category'] = self._rule_based_categorize(row['description'])
        
        return df
    
    def _rule_based_categorize(self, description: str) -> str:
        """دسته‌بندی مبتنی بر قوانین"""
        rules = {
            "درآمد": ["حقوق", "سود", "فروش", "درآمد", "واریز", "deposit", "income"],
            "خرید": ["خرید", "دیجی‌کالا", "فروشگاه", "shop", "purchase"],
            "قبوض": ["قبض", "برق", "آب", "گاز", "bill", "utility"],
            "حمل و نقل": ["بنزین", "تاکسی", "مترو", "اتوبوس", "gas", "taxi"],
            "غذا": ["سوپرمارکت", "رستوران", "غذا", "restaurant", "food"],
            "سرمایه گذاری": ["بورس", "سهام", "سرمایه", "stock", "investment"],
            "سلامت": ["درمان", "دارو", "بیمارستان", "health", "medical"],
            "آموزش": ["کتاب", "دوره", "آموزش", "book", "course"],
            "تفریح": ["سینما", "تفریح", "تئاتر", "cinema", "entertainment"]
        }
        
        desc_lower = str(description).lower()
        for category, keywords in rules.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return category
        
        return "سایر"
    
    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        تشخیص الگوهای مختلف در تراکنش‌ها
        
        Args:
            df (pd.DataFrame): دیتافریم پردازش شده
            
        Returns:
            pd.DataFrame: دیتافریم با ستون‌های الگوهای تشخیص داده شده
        """
        df = df.copy()
        
        # تشخیص تراکنش‌های تکراری
        df = self._detect_recurring_transactions(df)
        
        # تشخیص اشتراک‌ها
        df = self._detect_subscriptions(df)
        
        # تشخیص هزینه‌های غیرمعمول
        df = self._detect_unusual_transactions(df)
        
        return df
    
    def _detect_recurring_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """تشخیص تراکنش‌های تکراری"""
        df['is_recurring'] = False
        
        if 'description' not in df.columns:
            return df
            
        # گروه‌بندی بر اساس توضیحات
        desc_counts = df['description'].value_counts()
        recurring_descs = desc_counts[desc_counts > 1].index
        
        df.loc[df['description'].isin(recurring_descs), 'is_recurring'] = True
        
        # تشخیص الگوهای ماهانه (مثلاً حقوق ماهانه)
        for desc in recurring_descs:
            desc_data = df[df['description'] == desc].sort_values('date')
            if len(desc_data) >= 3:
                # بررسی فاصله زمانی بین تراکنش‌ها
                intervals = desc_data['date'].diff().dropna()
                avg_interval = intervals.mean()
                
                # اگر میانگین فاصله حدود 30 روز باشد
                if 25 <= avg_interval.days <= 35:
                    df.loc[df['description'] == desc, 'is_recurring'] = True
        
        return df
    
    def _detect_subscriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """تشخیص اشتراک‌های ماهانه/سالانه"""
        df['is_subscription'] = False
        
        if 'description' not in df.columns:
            return df
            
        subscription_keywords = [
            'اشتراک', 'subscription', 'month', 'yearly', 'monthly',
            'نفتال', 'netflix', 'spotify', '会员', '订阅'
        ]
        
        pattern = '|'.join(subscription_keywords)
        mask = df['description'].str.contains(pattern, case=False, na=False)
        df.loc[mask, 'is_subscription'] = True
        
        return df
    
    def _detect_unusual_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """تشخیص هزینه‌های غیرمعمول با روش IQR"""
        df['is_unusual'] = False
        
        if 'category' not in df.columns or 'amount_usd' not in df.columns:
            return df
            
        # فقط برای هزینه‌ها
        expense_df = df[df['type'] == 'هزینه'].copy()
        
        for category in expense_df['category'].unique():
            cat_data = expense_df[expense_df['category'] == category]
            
            if len(cat_data) < 3:
                continue
                
            # محاسبه IQR
            q1 = cat_data['amount_usd'].quantile(0.25)
            q3 = cat_data['amount_usd'].quantile(0.75)
            iqr = q3 - q1
            
            # محاسبه کران‌ها
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # شناسایی تراکنش‌های غیرمعمول
            unusual_mask = (
                (df['category'] == category) &
                (df['amount_usd'] > upper_bound) &
                (df['type'] == 'هزینه')
            )
            
            df.loc[unusual_mask, 'is_unusual'] = True
        
        return df
    
    def generate_insights(self, df: pd.DataFrame) -> Dict:
        """
        تولید بینش‌های مالی کلیدی از داده‌ها
        
        Args:
            df (pd.DataFrame): دیتافریم کامل
            
        Returns:
            Dict: دیکشنری شامل بینش‌های مالی
        """
        insights = {}
        
        # محاسبه درآمدها و هزینه‌ها
        income_df = df[df['type'] == 'درآمد']
        expense_df = df[df['type'] == 'هزینه']
        
        insights['total_income'] = income_df['amount_usd'].sum()
        insights['total_expense'] = abs(expense_df['amount_usd'].sum())
        insights['net_balance'] = insights['total_income'] - insights['total_expense']
        
        # محاسبه میانگین‌های ماهانه
        months_count = df['month'].nunique()
        insights['avg_monthly_income'] = insights['total_income'] / months_count if months_count > 0 else 0
        insights['avg_monthly_expense'] = insights['total_expense'] / months_count if months_count > 0 else 0
        
        # محاسبه نرخ پس‌انداز
        insights['savings_rate'] = (
            insights['net_balance'] / insights['total_income'] 
            if insights['total_income'] > 0 else 0
        )
        
        # بیشترین دسته‌بندی هزینه
        if not expense_df.empty:
            top_category = expense_df.groupby('category')['amount_usd'].sum().idxmax()
            insights['top_category'] = top_category
        else:
            insights['top_category'] = 'N/A'
        
        # شمارش الگوهای تشخیص داده شده
        insights['recurring_count'] = df['is_recurring'].sum()
        insights['subscription_count'] = df['is_subscription'].sum()
        insights['unusual_count'] = df['is_unusual'].sum()
        
        # محاسبه شاخص‌های پیشرفته
        insights = self._calculate_advanced_metrics(df, insights)
        
        return insights
    
    def _calculate_advanced_metrics(self, df: pd.DataFrame, insights: Dict) -> Dict:
        """محاسبه شاخص‌های مالی پیشرفته"""
        # نسبت هزینه به درآمد
        if insights['total_income'] > 0:
            insights['expense_to_income_ratio'] = insights['total_expense'] / insights['total_income']
        else:
            insights['expense_to_income_ratio'] = 0
        
        # نوسانات ماهانه
        monthly_data = df.groupby(['year', 'month'])['amount_usd'].sum()
        insights['monthly_volatility'] = monthly_data.std()
        
        # بزرگترین تراکنش
        if not df.empty:
            max_transaction = df.loc[df['amount_usd'].abs().idxmax()]
            insights['largest_transaction'] = {
                'description': max_transaction['description'],
                'amount': max_transaction['amount_usd'],
                'date': max_transaction['date'].strftime('%Y-%m-%d'),
                'type': max_transaction['type']
            }
        else:
            insights['largest_transaction'] = None
        
        # روندهای فصلی
        if 'quarter' in df.columns:
            quarterly_data = df.groupby('quarter')['amount_usd'].sum()
            insights['quarterly_trend'] = quarterly_data.to_dict()
        
        return insights
    
    def get_category_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        تحلیل دقیق هر دسته‌بندی
        
        Args:
            df (pd.DataFrame): دیتافریم کامل
            
        Returns:
            pd.DataFrame: تحلیل هر دسته‌بندی
        """
        category_analysis = df.groupby('category').agg(
            total_amount=('amount_usd', 'sum'),
            avg_amount=('amount_usd', 'mean'),
            transaction_count=('amount_usd', 'count'),
            min_amount=('amount_usd', 'min'),
            max_amount=('amount_usd', 'max')
        ).reset_index()
        
        # محاسبه سهم هر دسته از کل
        total_expense = abs(df[df['type'] == 'هزینه']['amount_usd'].sum())
        category_analysis['percentage'] = (
            category_analysis['total_amount'].abs() / total_expense * 100
        ).fillna(0)
        
        return category_analysis.sort_values('total_amount', ascending=False)
    
    def get_monthly_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        محاسبه روند مالی ماهانه
        
        Args:
            df (pd.DataFrame): دیتافریم کامل
            
        Returns:
            pd.DataFrame: روند ماهانه
        """
        monthly_trend = df.groupby(['year', 'month']).agg(
            income=('amount_usd', lambda x: x[x > 0].sum()),
            expense=('amount_usd', lambda x: abs(x[x < 0].sum())),
            net=('amount_usd', 'sum')
        ).reset_index()
        
        # ایجاد ستون تاریخ برای نمودار
        monthly_trend['date'] = pd.to_datetime(
            monthly_trend['year'].astype(str) + '-' + 
            monthly_trend['month'].astype(str) + '-01'
        )
        
        return monthly_trend
