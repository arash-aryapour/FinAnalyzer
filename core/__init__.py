# core/__init__.py

"""
Core module for FinAnalyzer - Advanced Financial Analysis Tool

This module provides the main components for financial data processing,
machine learning-based transaction categorization, and report generation.

Author: Arash Aryapour
Version: 1.0.0
License: MIT
"""

# Import main classes and functions
from .data_processor import FinancialDataProcessor
from .ml_engine import TransactionCategorizer
from .report_generator import generate_financial_report

# Package metadata
__version__ = "1.0.0"
__author__ = "Arash Aryapour"
__email__ = "arash.aryapour@example.com"
__description__ = "Advanced financial analysis engine with AI capabilities"

# Define what gets imported with "from core import *"
__all__ = [
    "FinancialDataProcessor",
    "TransactionCategorizer", 
    "generate_financial_report",
    "__version__",
    "__author__"
]

# Initialize configuration (if needed)
DEFAULT_CONFIG = {
    "currency_base": "USD",
    "categorization_method": "hybrid",
    "report_format": "pdf",
    "language": "fa"
}

# Package-level utilities
def get_version():
    """Return the current version of the core module"""
    return __version__

def get_supported_currencies():
    """Return list of supported currencies"""
    return ["USD", "EUR", "GBP", "IRR", "BTC", "ETH"]

def get_supported_categories():
    """Return list of supported transaction categories"""
    return [
        "درآمد", "خرید", "قبوض", "حمل و نقل", "غذا",
        "سرمایه گذاری", "سلامت", "آموزش", "تفریح", "سایر"
    ]

# Module initialization message (for debugging)
if __debug__:
    print(f"FinAnalyzer Core v{__version__} initialized")
