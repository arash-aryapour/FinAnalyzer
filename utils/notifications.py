# utils/notifications.py

import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional, Union
import os
import logging
from datetime import datetime
import time
import threading
from jinja2 import Template
import warnings
warnings.filterwarnings('ignore')

class NotificationManager:
    """
    مدیر اعلان‌ها برای ارسال پیام‌های مالی از کانال‌های مختلف
    شامل قابلیت‌های:
    - ارسال ایمیل با قالب‌های HTML
    - ارسال پیام به تلگرام
    - ارسال وب‌هوک
    - مدیریت قالب‌های پیام
    - سیستم صف و تلاش مجدد
    - لاگ‌گیری کامل
    """
    
    def __init__(self, config: Dict = None):
        """
        مقداردهی اولیه مدیر اعلان‌ها
        
        Args:
            config (Dict): تنظیمات شامل:
                - email: تنظیمات SMTP
                - telegram: تنظیمات ربات تلگرام
                - webhook: تنظیمات وب‌هوک
                - templates: مسیر قالب‌های پیام
        """
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # تنظیمات پیش‌فرض
        self.default_channel = self.config.get('default_channel', 'email')
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 5)  # ثانیه
        
        # بارگذاری تنظیمات کانال‌ها
        self.email_config = self.config.get('email', {})
        self.telegram_config = self.config.get('telegram', {})
        self.webhook_config = self.config.get('webhook', {})
        
        # بارگذاری قالب‌ها
        self.templates = self._load_templates()
        
        # صف اعلان‌ها
        self.notification_queue = []
        self.queue_lock = threading.Lock()
        
        # شروع پردازشگر صف
        self._start_queue_processor()
    
    def _setup_logger(self) -> logging.Logger:
        """راه‌اندازی سیستم لاگ‌گیری"""
        logger = logging.getLogger('NotificationManager')
        logger.setLevel(logging.INFO)
        
        # ایجاد فرمت لاگ
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # کنسول هندلر
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # فایل هندلر
        file_handler = logging.FileHandler('notifications.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_templates(self) -> Dict:
        """بارگذاری قالب‌های پیام"""
        template_dir = self.config.get('template_dir', 'templates/notifications')
        templates = {}
        
        if os.path.exists(template_dir):
            for filename in os.listdir(template_dir):
                if filename.endswith('.html'):
                    template_name = filename[:-5]  # حذف پسوند .html
                    with open(os.path.join(template_dir, filename), 'r', encoding='utf-8') as f:
                        templates[template_name] = Template(f.read())
        
        # قالب‌های پیش‌فرض
        if not templates:
            templates = self._get_default_templates()
        
        return templates
    
    def _get_default_templates(self) -> Dict:
        """قالب‌های پیش‌فرض پیام‌ها"""
        return {
            'financial_summary': Template('''
                <h2>خلاصه مالی شما</h2>
                <p>تاریخ: {{ date }}</p>
                <ul>
                    <li>درآمد کل: {{ income }}</li>
                    <li>هزینه کل: {{ expense }}</li>
                    <li>تراز خالص: {{ balance }}</li>
                    <li>نرخ پس‌انداز: {{ savings_rate }}%</li>
                </ul>
            '''),
            'alert': Template('''
                <h3>هشدار مالی</h3>
                <p>{{ message }}</p>
                <p>زمان: {{ time }}</p>
            '''),
            'report_ready': Template('''
                <h3>گزارش مالی شما آماده است</h3>
                <p>گزارش {{ report_type }} در تاریخ {{ date }} تولید شد.</p>
                <p>می‌توانید آن را از پنل کاربری خود دانلود کنید.</p>
            ''')
        }
    
    def send_notification(self, 
                         message: str, 
                         channel: str = None,
                         template: str = None,
                         template_data: Dict = None,
                         attachments: List[str] = None,
                         priority: str = 'normal') -> bool:
        """
        ارسال اعلان از طریق کانال مشخص شده
        
        Args:
            message (str): متن پیام
            channel (str): کانال ارسال (email, telegram, webhook)
            template (str): نام قالب پیام
            template_data (Dict): داده‌های قالب
            attachments (List[str]): لیست فایل‌های پیوست
            priority (str): اولویت (high, normal, low)
            
        Returns:
            bool: موفقیت‌آمیز بودن ارسال
        """
        channel = channel or self.default_channel
        
        # اعتبارسنجی کانال
        if channel not in ['email', 'telegram', 'webhook']:
            self.logger.error(f"Unsupported notification channel: {channel}")
            return False
        
        # آماده‌سازی پیام
        if template and template in self.templates:
            template_data = template_data or {}
            template_data['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = self.templates[template].render(**template_data)
        
        # ایجاد اعلان
        notification = {
            'message': message,
            'channel': channel,
            'attachments': attachments or [],
            'priority': priority,
            'attempts': 0,
            'created_at': datetime.now()
        }
        
        # افزودن به صف
        with self.queue_lock:
            self.notification_queue.append(notification)
        
        self.logger.info(f"Notification queued for {channel} channel")
        return True
    
    def _start_queue_processor(self):
        """شروع پردازشگر صف در یک ترد جداگانه"""
        def process_queue():
            while True:
                with self.queue_lock:
                    if self.notification_queue:
                        notification = self.notification_queue.pop(0)
                    else:
                        notification = None
                
                if notification:
                    self._process_notification(notification)
                else:
                    time.sleep(1)  # انتظار برای اعلان جدید
        
        processor_thread = threading.Thread(target=process_queue, daemon=True)
        processor_thread.start()
    
    def _process_notification(self, notification: Dict):
        """پردازش یک اعلان از صف"""
        channel = notification['channel']
        
        for attempt in range(self.max_retries):
            try:
                if channel == 'email':
                    success = self._send_email(notification)
                elif channel == 'telegram':
                    success = self._send_telegram(notification)
                elif channel == 'webhook':
                    success = self._send_webhook(notification)
                else:
                    success = False
                
                if success:
                    self.logger.info(f"Notification sent successfully via {channel}")
                    return
                
                # تلاش مجدد در صورت خطا
                time.sleep(self.retry_delay)
                
            except Exception as e:
                self.logger.error(f"Error sending notification (attempt {attempt + 1}): {str(e)}")
        
        self.logger.error(f"Failed to send notification after {self.max_retries} attempts")
    
    def _send_email(self, notification: Dict) -> bool:
        """ارسال ایمیل"""
        config = self.email_config
        
        if not all([config.get('smtp_server'), config.get('smtp_port'), 
                   config.get('username'), config.get('password')]):
            self.logger.error("Email configuration incomplete")
            return False
        
        # ایجاد پیام
        msg = MIMEMultipart('alternative')
        msg['Subject'] = notification.get('subject', 'اعلان مالی')
        msg['From'] = config['username']
        msg['To'] = config.get('to', config['username'])
        
        # افزودن متن HTML
        html_part = MIMEText(notification['message'], 'html', 'utf-8')
        msg.attach(html_part)
        
        # افزودن پیوست‌ها
        for attachment_path in notification['attachments']:
            if os.path.exists(attachment_path):
                with open(attachment_path, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {os.path.basename(attachment_path)}'
                    )
                    msg.attach(part)
        
        # ارسال ایمیل
        try:
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['username'], config['password'])
                server.send_message(msg)
            return True
        except Exception as e:
            self.logger.error(f"Email sending failed: {str(e)}")
            return False
    
    def _send_telegram(self, notification: Dict) -> bool:
        """ارسال پیام به تلگرام"""
        config = self.telegram_config
        
        if not config.get('bot_token') or not config.get('chat_id'):
            self.logger.error("Telegram configuration incomplete")
            return False
        
        # آماده‌سازی پیام
        text = notification['message']
        
        # محدودیت طول پیام در تلگرام
        if len(text) > 4096:
            text = text[:4093] + "..."
        
        # ارسال پیام
        try:
            url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
            data = {
                'chat_id': config['chat_id'],
                'text': text,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            # ارسال پیوست‌ها
            for attachment_path in notification['attachments']:
                self._send_telegram_document(attachment_path)
            
            return True
        except Exception as e:
            self.logger.error(f"Telegram sending failed: {str(e)}")
            return False
    
    def _send_telegram_document(self, file_path: str) -> bool:
        """ارسال فایل به تلگرام"""
        config = self.telegram_config
        
        try:
            url = f"https://api.telegram.org/bot{config['bot_token']}/sendDocument"
            
            with open(file_path, 'rb') as file:
                files = {'document': file}
                data = {'chat_id': config['chat_id']}
                
                response = requests.post(url, files=files, data=data, timeout=30)
                response.raise_for_status()
            
            return True
        except Exception as e:
            self.logger.error(f"Telegram document sending failed: {str(e)}")
            return False
    
    def _send_webhook(self, notification: Dict) -> bool:
        """ارسال وب‌هوک"""
        config = self.webhook_config
        
        if not config.get('url'):
            self.logger.error("Webhook URL not configured")
            return False
        
        # آماده‌سازی داده‌ها
        payload = {
            'message': notification['message'],
            'timestamp': datetime.now().isoformat(),
            'source': 'FinAnalyzer',
            'priority': notification['priority']
        }
        
        # افزودن پیوست‌ها
        if notification['attachments']:
            payload['attachments'] = notification['attachments']
        
        # ارسال درخواست
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'FinAnalyzer/1.0'
            }
            
            # افزودن هدرهای سفارشی
            if config.get('headers'):
                headers.update(config['headers'])
            
            response = requests.post(
                config['url'],
                json=payload,
                headers=headers,
                timeout=config.get('timeout', 30)
            )
            response.raise_for_status()
            
            return True
        except Exception as e:
            self.logger.error(f"Webhook sending failed: {str(e)}")
            return False
    
    def send_financial_summary(self, insights: Dict, channel: str = None) -> bool:
        """ارسال خلاصه مالی"""
        template_data = {
            'income': f"{insights['total_income']:,.0f} $",
            'expense': f"{insights['total_expense']:,.0f} $",
            'balance': f"{insights['net_balance']:,.0f} $",
            'savings_rate': f"{insights['savings_rate']:.1%}"
        }
        
        return self.send_notification(
            message='',
            channel=channel,
            template='financial_summary',
            template_data=template_data
        )
    
    def send_alert(self, message: str, alert_type: str = 'warning', channel: str = None) -> bool:
        """ارسال هشدار"""
        template_data = {
            'message': message,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': alert_type
        }
        
        return self.send_notification(
            message='',
            channel=channel,
            template='alert',
            template_data=template_data,
            priority='high'
        )
    
    def send_report_ready(self, report_type: str, channel: str = None) -> bool:
        """اعلام آماده بودن گزارش"""
        template_data = {
            'report_type': report_type,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        return self.send_notification(
            message='',
            channel=channel,
            template='report_ready',
            template_data=template_data
        )
    
    def test_configuration(self) -> Dict[str, bool]:
        """تست تنظیمات تمام کانال‌ها"""
        results = {}
        
        # تست ایمیل
        try:
            results['email'] = self._send_email({
                'message': 'Test email from FinAnalyzer',
                'subject': 'Test Notification',
                'attachments': []
            })
        except:
            results['email'] = False
        
        # تست تلگرام
        try:
            results['telegram'] = self._send_telegram({
                'message': 'Test message from FinAnalyzer',
                'attachments': []
            })
        except:
            results['telegram'] = False
        
        # تست وب‌هوک
        try:
            results['webhook'] = self._send_webhook({
                'message': 'Test webhook from FinAnalyzer',
                'attachments': []
            })
        except:
            results['webhook'] = False
        
        return results
    
    def get_queue_status(self) -> Dict:
        """دریافت وضعیت صف اعلان‌ها"""
        with self.queue_lock:
            queue_length = len(self.notification_queue)
        
        return {
            'queue_length': queue_length,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'last_activity': datetime.now().isoformat()
        }
    
    def clear_queue(self):
        """پاک کردن صف اعلان‌ها"""
        with self.queue_lock:
            self.notification_queue.clear()
        self.logger.info("Notification queue cleared")
    
    def add_template(self, name: str, template_content: str):
        """افزودن قالب جدید"""
        self.templates[name] = Template(template_content)
        self.logger.info(f"Added new template: {name}")
    
    def get_supported_channels(self) -> List[str]:
        """دریافت لیست کانال‌های پشتیبانی شده"""
        return ['email', 'telegram', 'webhook']

# تابع کمکی برای مقداردهی اولیه
def initialize_notification_manager(config: Dict = None) -> NotificationManager:
    """
    مقداردهی اولیه مدیر اعلان‌ها با تنظیمات سفارشی
    
    Args:
        config (Dict): تنظیمات سفارشی
        
    Returns:
        NotificationManager: نمونه‌ای از مدیر اعلان‌ها
    """
    return NotificationManager(config)
