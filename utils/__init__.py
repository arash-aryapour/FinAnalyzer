# utils/__init__.py

"""
ماژول ابزارهای کمکی برای FinAnalyzer

این ماژول شامل توابع و کلاس‌های کمکی برای عملیات‌های جانبی مانند:
- تبدیل ارز
- سیستم اعلان‌ها
- پردازش متن
- ابزارهای ریاضی و آماری
- مدیریت فایل‌ها
- ابزارهای شبکه و API

Author: Arash Aryapour
Version: 1.0.0
License: MIT
"""

# Import utility modules
from .currency import CurrencyConverter
from .notifications import NotificationManager
from .text_utils import TextProcessor
from .math_utils import FinancialMath
from .file_utils import FileManager
from .api_utils import APIManager

# Package metadata
__version__ = "1.0.0"
__author__ = "Arash Aryapour"
__email__ = "arash.aryapour@example.com"
__description__ = "Utility tools for FinAnalyzer financial analysis platform"

# Define what gets imported with "from utils import *"
__all__ = [
    "CurrencyConverter",
    "NotificationManager", 
    "TextProcessor",
    "FinancialMath",
    "FileManager",
    "APIManager",
    "get_version",
    "get_supported_currencies",
    "get_notification_channels"
]

# Package-level configuration
DEFAULT_CONFIG = {
    "api_timeout": 30,
    "max_retries": 3,
    "currency_api": "https://api.exchangerate-api.com/v4/latest/USD",
    "notification_channels": ["email", "telegram", "webhook"],
    "file_upload_max_size": 50 * 1024 * 1024,  # 50MB
    "text_encoding": "utf-8"
}

# Package-level utility functions
def get_version():
    """برگرداندن نسخه فعلی ماژول ابزارها"""
    return __version__

def get_supported_currencies():
    """لیست ارزهای پشتیبانی شده را برمی‌گرداند"""
    return [
        "USD", "EUR", "GBP", "JPY", "CNY", "AUD", "CAD", "CHF",
        "IRR", "TRY", "RUB", "INR", "BRL", "MXN", "ZAR",
        "BTC", "ETH", "BNB", "ADA", "DOT", "XRP"
    ]

def get_notification_channels():
    """لیست کانال‌های اطلاع‌رسانی پشتیبانی شده را برمی‌گرداند"""
    return DEFAULT_CONFIG["notification_channels"]

def initialize_utils(config=None):
    """مقداردهی اولیه ماژول ابزارها با تنظیمات سفارشی"""
    global DEFAULT_CONFIG
    
    if config:
        DEFAULT_CONFIG.update(config)
    
    # مقداردهی اولیه زیرماژول‌ها
    CurrencyConverter.initialize(DEFAULT_CONFIG)
    NotificationManager.initialize(DEFAULT_CONFIG)
    APIManager.initialize(DEFAULT_CONFIG)
    
    if __debug__:
        print(f"FinAnalyzer Utils v{__version__} initialized with custom config")

# Error classes for better error handling
class UtilsError(Exception):
    """پایه کلاس خطا برای ماژول ابزارها"""
    pass

class CurrencyConversionError(UtilsError):
    """خطا در تبدیل ارز"""
    pass

class NotificationError(UtilsError):
    """خطا در ارسال اعلان"""
    pass

class FileProcessingError(UtilsError):
    """خطا در پردازش فایل"""
    pass

class APIError(UtilsError):
    """خطا در ارتباط با API"""
    pass

# Validation functions
def validate_currency(currency):
    """اعتبارسنجی کد ارز"""
    if currency not in get_supported_currencies():
        raise CurrencyConversionError(f"Unsupported currency: {currency}")
    return True

def validate_notification_channel(channel):
    """اعتبارسنجی کانال اطلاع‌رسانی"""
    if channel not in get_notification_channels():
        raise NotificationError(f"Unsupported notification channel: {channel}")
    return True

def validate_file_size(file_size):
    """اعتبارسنجی اندازه فایل"""
    max_size = DEFAULT_CONFIG["file_upload_max_size"]
    if file_size > max_size:
        raise FileProcessingError(f"File size exceeds maximum limit of {max_size/1024/1024}MB")
    return True

# Utility decorators
def handle_errors(func):
    """دکوراتور برای مدیریت خطاها"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except UtilsError as e:
            print(f"Utils Error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error in {func.__name__}: {str(e)}")
            raise UtilsError(f"Unexpected error in {func.__name__}")
    return wrapper

def log_execution(func):
    """دکوراتور برای لاگ کردن اجرای توابع"""
    def wrapper(*args, **kwargs):
        if __debug__:
            print(f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        if __debug__:
            print(f"Completed {func.__name__} with result: {result}")
        return result
    return wrapper

# Package initialization
if __debug__:
    print(f"FinAnalyzer Utils v{__version__} loaded successfully")
    print(f"Supported currencies: {len(get_supported_currencies())}")
    print(f"Notification channels: {get_notification_channels()}")
