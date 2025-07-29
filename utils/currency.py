# utils/currency.py

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Union
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class CurrencyConverter:
    """
    کلاس تبدیل ارز با پشتیبانی از ارزهای فیات و دیجیتال
    شامل قابلیت‌های:
    - تبدیل آنی ارزها
    - دریافت نرخ‌های لحظه‌ای
    - کش کردن نرخ‌ها برای بهبود عملکرد
    - پشتیبانی از آفلاین
    - تبدیل چند ارزی همزمان
    """
    
    def __init__(self, config: Dict = None):
        """
        مقداردهی اولیه مبدل ارز
        
        Args:
            config (Dict): تنظیمات سفارشی شامل:
                - api_key: کلید API برای سرویس‌های ارزی
                - base_currency: ارز پایه (پیش‌فرض USD)
                - cache_duration: مدت زمان کش به دقیقه (پیش‌فرض 60)
                - api_url: آدرس API سرویس ارز
        """
        self.config = config or {}
        self.api_key = self.config.get('api_key', None)
        self.base_currency = self.config.get('base_currency', 'USD')
        self.cache_duration = self.config.get('cache_duration', 60)  # دقیقه
        
        # آدرس‌های API پیش‌فرض
        self.fiat_api_url = self.config.get('fiat_api_url', 'https://api.exchangerate-api.com/v4/latest/')
        self.crypto_api_url = self.config.get('crypto_api_url', 'https://api.coingecko.com/api/v3')
        
        # کش نرخ‌ها
        self.fiat_rates = {}
        self.crypto_rates = {}
        self.last_update = None
        
        # بارگذاری نرخ‌های اولیه
        self._load_initial_rates()
    
    def _load_initial_rates(self):
        """بارگذاری نرخ‌های اولیه"""
        try:
            # تلاش برای بارگذاری از کش محلی
            self._load_from_cache()
            
            # اگر کش منقضی شده یا وجود ندارد، به‌روزرسانی کن
            if self._is_cache_expired():
                self.update_all_rates()
        except Exception as e:
            print(f"Error loading initial rates: {e}")
            # استفاده از نرخ‌های پیش‌فرض در صورت خطا
            self._load_default_rates()
    
    def _load_from_cache(self):
        """بارگذاری نرخ‌ها از کش محلی"""
        cache_file = 'currency_cache.json'
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.fiat_rates = cache_data.get('fiat_rates', {})
                    self.crypto_rates = cache_data.get('crypto_rates', {})
                    self.last_update = datetime.fromisoformat(cache_data.get('last_update'))
            except Exception as e:
                print(f"Error loading cache: {e}")
    
    def _save_to_cache(self):
        """ذخیره نرخ‌ها در کش محلی"""
        cache_data = {
            'fiat_rates': self.fiat_rates,
            'crypto_rates': self.crypto_rates,
            'last_update': datetime.now().isoformat()
        }
        
        try:
            with open('currency_cache.json', 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _is_cache_expired(self) -> bool:
        """بررسی انقضای کش"""
        if not self.last_update:
            return True
        
        expiration_time = self.last_update + timedelta(minutes=self.cache_duration)
        return datetime.now() > expiration_time
    
    def _load_default_rates(self):
        """بارگذاری نرخ‌های پیش‌فرض"""
        # نرخ‌های پیش‌فرض برای ارزهای اصلی
        self.fiat_rates = {
            'USD': 1.0,
            'EUR': 0.85,
            'GBP': 0.73,
            'JPY': 110.0,
            'IRR': 42000.0,
            'TRY': 8.5,
            'CNY': 6.5,
            'RUB': 75.0
        }
        
        # نرخ‌های پیش‌فرض برای ارزهای دیجیتال
        self.crypto_rates = {
            'BTC': 45000.0,
            'ETH': 3000.0,
            'BNB': 400.0,
            'ADA': 1.2,
            'DOT': 25.0,
            'XRP': 1.0
        }
        
        self.last_update = datetime.now()
    
    def update_all_rates(self):
        """به‌روزرسانی تمام نرخ‌ها"""
        try:
            self._update_fiat_rates()
            self._update_crypto_rates()
            self._save_to_cache()
            self.last_update = datetime.now()
            print("Currency rates updated successfully")
        except Exception as e:
            print(f"Error updating rates: {e}")
            raise
    
    def _update_fiat_rates(self):
        """به‌روزرسانی نرخ‌های ارزهای فیات"""
        try:
            url = f"{self.fiat_api_url}{self.base_currency}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.fiat_rates = data.get('rates', {})
            
            # اضافه کردن ارز پایه
            self.fiat_rates[self.base_currency] = 1.0
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating fiat rates: {e}")
            raise
    
    def _update_crypto_rates(self):
        """به‌روزرسانی نرخ‌های ارزهای دیجیتال"""
        try:
            # دریافت نرخ‌های ارزهای دیجیتال نسبت به USD
            params = {
                'ids': 'bitcoin,ethereum,binancecoin,cardano,polkadot,ripple',
                'vs_currencies': 'usd'
            }
            
            response = requests.get(
                f"{self.crypto_api_url}/simple/price",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # تبدیل به فرمت استاندارد
            self.crypto_rates = {
                'BTC': data['bitcoin']['usd'],
                'ETH': data['ethereum']['usd'],
                'BNB': data['binancecoin']['usd'],
                'ADA': data['cardano']['usd'],
                'DOT': data['polkadot']['usd'],
                'XRP': data['ripple']['usd']
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error updating crypto rates: {e}")
            raise
    
    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """
        تبدیل مبلغ از یک ارز به ارز دیگر
        
        Args:
            amount (float): مبلغ برای تبدیل
            from_currency (str): ارز مبدا
            to_currency (str): ارز مقصد
            
        Returns:
            float: مبلغ تبدیل شده
        """
        if amount == 0:
            return 0.0
        
        # بررسی انقضای کش
        if self._is_cache_expired():
            try:
                self.update_all_rates()
            except:
                print("Using cached rates due to update failure")
        
        # تبدیل ارزهای فیات
        if from_currency in self.fiat_rates and to_currency in self.fiat_rates:
            return self._convert_fiat(amount, from_currency, to_currency)
        
        # تبدیل ارزهای دیجیتال
        elif from_currency in self.crypto_rates and to_currency in self.crypto_rates:
            return self._convert_crypto(amount, from_currency, to_currency)
        
        # تبدیل ترکیبی (فیات به دیجیتال و برعکس)
        elif from_currency in self.fiat_rates and to_currency in self.crypto_rates:
            return self._convert_fiat_to_crypto(amount, from_currency, to_currency)
        
        elif from_currency in self.crypto_rates and to_currency in self.fiat_rates:
            return self._convert_crypto_to_fiat(amount, from_currency, to_currency)
        
        else:
            raise ValueError(f"Unsupported currency pair: {from_currency} -> {to_currency}")
    
    def _convert_fiat(self, amount: float, from_currency: str, to_currency: str) -> float:
        """تبدیل بین ارزهای فیات"""
        if from_currency == to_currency:
            return amount
        
        # تبدیل به ارز پایه سپس به ارز مقصد
        amount_in_base = amount / self.fiat_rates[from_currency]
        return amount_in_base * self.fiat_rates[to_currency]
    
    def _convert_crypto(self, amount: float, from_currency: str, to_currency: str) -> float:
        """تبدیل بین ارزهای دیجیتال"""
        if from_currency == to_currency:
            return amount
        
        # تبدیل به USD سپس به ارز مقصد
        amount_in_usd = amount * self.crypto_rates[from_currency]
        return amount_in_usd / self.crypto_rates[to_currency]
    
    def _convert_fiat_to_crypto(self, amount: float, from_currency: str, to_currency: str) -> float:
        """تبدیل ارز فیات به دیجیتال"""
        amount_in_usd = self._convert_fiat(amount, from_currency, 'USD')
        return amount_in_usd / self.crypto_rates[to_currency]
    
    def _convert_crypto_to_fiat(self, amount: float, from_currency: str, to_currency: str) -> float:
        """تبدیل ارز دیجیتال به فیات"""
        amount_in_usd = amount * self.crypto_rates[from_currency]
        return self._convert_fiat(amount_in_usd, 'USD', to_currency)
    
    def get_rate(self, from_currency: str, to_currency: str = None) -> float:
        """
        دریافت نرخ تبدیل بین دو ارز
        
        Args:
            from_currency (str): ارز مبدا
            to_currency (str): ارز مقصد (پیش‌فرض: ارز پایه)
            
        Returns:
            float: نرخ تبدیل
        """
        to_currency = to_currency or self.base_currency
        return self.convert(1.0, from_currency, to_currency)
    
    def get_supported_currencies(self) -> Dict[str, list]:
        """
        دریافت لیست ارزهای پشتیبانی شده
        
        Returns:
            Dict: دیکشنری شامل ارزهای فیات و دیجیتال
        """
        return {
            'fiat': list(self.fiat_rates.keys()),
            'crypto': list(self.crypto_rates.keys())
        }
    
    def convert_multiple(self, amount: float, from_currency: str, to_currencies: list) -> Dict[str, float]:
        """
        تبدیل یک مبلغ به چندین ارز به صورت همزمان
        
        Args:
            amount (float): مبلغ برای تبدیل
            from_currency (str): ارز مبدا
            to_currencies (list): لیست ارزهای مقصد
            
        Returns:
            Dict: دیکشنری ارزهای مقصد و مبالغ تبدیل شده
        """
        results = {}
        
        for currency in to_currencies:
            try:
                results[currency] = self.convert(amount, from_currency, currency)
            except ValueError:
                results[currency] = None
        
        return results
    
    def get_historical_rate(self, from_currency: str, to_currency: str, date: str) -> Optional[float]:
        """
        دریافت نرخ تاریخی ارز (نیاز به API پولی دارد)
        
        Args:
            from_currency (str): ارز مبدا
            to_currency (str): ارز مقصد
            date (str): تاریخ به فرمت YYYY-MM-DD
            
        Returns:
            Optional[float]: نرخ تاریخی یا None در صورت خطا
        """
        # این قابلیت نیاز به API پولی دارد
        # در اینجا فقط پیاده‌سازی پایه ارائه می‌شود
        try:
            url = f"https://api.exchangerate.host/{date}"
            params = {
                'base': from_currency,
                'symbols': to_currency
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data['rates'].get(to_currency)
            
        except Exception as e:
            print(f"Error getting historical rate: {e}")
            return None
    
    def get_currency_info(self, currency: str) -> Dict:
        """
        دریافت اطلاعات کامل یک ارز
        
        Args:
            currency (str): کد ارز
            
        Returns:
            Dict: اطلاعات ارز
        """
        info = {
            'code': currency,
            'type': None,
            'rate_vs_base': None,
            'name': None,
            'symbol': None
        }
        
        if currency in self.fiat_rates:
            info['type'] = 'fiat'
            info['rate_vs_base'] = self.fiat_rates[currency]
            info['name'] = self._get_fiat_name(currency)
            info['symbol'] = self._get_fiat_symbol(currency)
        
        elif currency in self.crypto_rates:
            info['type'] = 'crypto'
            info['rate_vs_base'] = self.crypto_rates[currency]
            info['name'] = self._get_crypto_name(currency)
            info['symbol'] = self._get_crypto_symbol(currency)
        
        return info
    
    def _get_fiat_name(self, currency: str) -> str:
        """نام کامل ارز فیات"""
        names = {
            'USD': 'دلار آمریکا',
            'EUR': 'یورو',
            'GBP': 'پوند بریتانیا',
            'JPY': 'ین ژاپن',
            'IRR': 'ریال ایران',
            'TRY': 'لیر ترکیه',
            'CNY': 'یوآن چین',
            'RUB': 'روبل روسیه'
        }
        return names.get(currency, currency)
    
    def _get_fiat_symbol(self, currency: str) -> str:
        """نماد ارز فیات"""
        symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'IRR': 'ریال',
            'TRY': '₺',
            'CNY': '¥',
            'RUB': '₽'
        }
        return symbols.get(currency, currency)
    
    def _get_crypto_name(self, currency: str) -> str:
        """نام کامل ارز دیجیتال"""
        names = {
            'BTC': 'بیت‌کوین',
            'ETH': 'اتریوم',
            'BNB': 'بایننس کوین',
            'ADA': 'کاردانو',
            'DOT': 'پولکادوت',
            'XRP': 'ریپل'
        }
        return names.get(currency, currency)
    
    def _get_crypto_symbol(self, currency: str) -> str:
        """نماد ارز دیجیتال"""
        symbols = {
            'BTC': '₿',
            'ETH': 'Ξ',
            'BNB': 'BNB',
            'ADA': 'ADA',
            'DOT': 'DOT',
            'XRP': 'XRP'
        }
        return symbols.get(currency, currency)
    
    def format_amount(self, amount: float, currency: str, include_symbol: bool = True) -> str:
        """
        قالب‌بندی مبلغ با ارز مربوطه
        
        Args:
            amount (float): مبلغ
            currency (str): کد ارز
            include_symbol (bool): افزودن نماد ارز
            
        Returns:
            str: مبلغ قالب‌بندی شده
        """
        # دریافت اطلاعات ارز
        info = self.get_currency_info(currency)
        
        # قالب‌بندی عدد
        if info['type'] == 'crypto':
            # ارزهای دیجیتال معمولاً تا 8 رقم اعشار
            formatted = f"{amount:.8f}".rstrip('0').rstrip('.')
        else:
            # ارزهای فیات معمولاً تا 2 رقم اعشار
            formatted = f"{amount:,.2f}"
        
        # افزودن نماد
        if include_symbol and info['symbol']:
            if info['type'] == 'fiat':
                formatted = f"{info['symbol']}{formatted}"
            else:
                formatted = f"{formatted} {info['symbol']}"
        
        return formatted
    
    def get_rate_trends(self, currency: str, days: int = 7) -> pd.DataFrame:
        """
        دریافت روند تغییرات نرخ ارز در چند روز گذشته
        
        Args:
            currency (str): کد ارز
            days (int): تعداد روزها
            
        Returns:
            pd.DataFrame: داده‌های روند تغییرات
        """
        # این قابلیت نیاز به API پولی دارد
        # در اینجا فقط پیاده‌سازی پایه ارائه می‌شود
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # ایجاد داده‌های نمونه
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            base_rate = self.get_rate(currency, 'USD')
            
            # شبیه‌سازی نوسانات
            np.random.seed(42)
            changes = np.random.normal(0, 0.02, len(dates))
            rates = [base_rate * (1 + change) for change in changes]
            
            df = pd.DataFrame({
                'date': dates,
                'rate': rates,
                'currency': currency
            })
            
            return df
            
        except Exception as e:
            print(f"Error getting rate trends: {e}")
            return pd.DataFrame()
    
    def calculate_volatility(self, currency: str, days: int = 30) -> float:
        """
        محاسبه نوسانات ارز در یک دوره زمانی
        
        Args:
            currency (str): کد ارز
            days (int): تعداد روزها
            
        Returns:
            float: نوسانات (انحراف معیار)
        """
        df = self.get_rate_trends(currency, days)
        
        if df.empty:
            return 0.0
        
        returns = df['rate'].pct_change().dropna()
        return returns.std() * np.sqrt(365)  # نوسانات سالانه

# تابع کمکی برای مقداردهی اولیه
def initialize_currency_converter(config: Dict = None) -> CurrencyConverter:
    """
    مقداردهی اولیه مبدل ارز با تنظیمات سفارشی
    
    Args:
        config (Dict): تنظیمات سفارشی
        
    Returns:
        CurrencyConverter: نمونه‌ای از مبدل ارز
    """
    return CurrencyConverter(config)
