# core/report_generator.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
import os
from fpdf import FPDF, HTMLMixin
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# تنظیمات فونت برای پشتیبانی از فارسی
try:
    from bidi.algorithm import get_display
    import arabic_reshaper
    PERSIAN_SUPPORT = True
except ImportError:
    PERSIAN_SUPPORT = False

class FinancialReportGenerator:
    """
    کلاس اصلی برای تولید گزارش‌های مالی در فرمت‌های مختلف
    شامل PDF, HTML, Excel با قابلیت‌های شخصی‌سازی پیشرفته
    """
    
    def __init__(self, config: Dict = None):
        """
        مقداردهی اولیه تولیدکننده گزارش‌ها
        
        Args:
            config (Dict): تنظیمات سفارشی برای گزارش‌ها
        """
        self.config = config or {}
        self.template_dir = self.config.get('template_dir', 'templates')
        self.font_path = self.config.get('font_path', 'assets/fonts/Vazir.ttf')
        self.logo_path = self.config.get('logo_path', 'assets/logo.png')
        
        # تنظیمات پیش‌فرض
        self.default_language = self.config.get('language', 'fa')
        self.default_currency = self.config.get('currency', 'IRR')
        self.date_format = self.config.get('date_format', '%Y-%m-%d')
        
        # بارگذاری قالب‌ها
        self._load_templates()
    
    def _load_templates(self):
        """بارگذاری قالب‌های HTML"""
        if os.path.exists(self.template_dir):
            self.jinja_env = Environment(
                loader=FileSystemLoader(self.template_dir),
                autoescape=True
            )
        else:
            self.jinja_env = None
    
    def generate_report(self, 
                      data: Dict, 
                      report_type: str = 'pdf',
                      output_path: str = None,
                      template_name: str = None) -> str:
        """
        تولید گزارش در فرمت مورد نظر
        
        Args:
            data (Dict): داده‌های گزارش شامل insights, transactions, charts
            report_type (str): نوع گزارش (pdf, html, excel)
            output_path (str): مسیر ذخیره گزارش
            template_name (str): نام قالب سفارشی
            
        Returns:
            str: مسیر فایل تولید شده
        """
        if report_type == 'pdf':
            return self._generate_pdf_report(data, output_path, template_name)
        elif report_type == 'html':
            return self._generate_html_report(data, output_path, template_name)
        elif report_type == 'excel':
            return self._generate_excel_report(data, output_path)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
    
    def _generate_pdf_report(self, data: Dict, output_path: str = None, template_name: str = None) -> str:
        """تولید گزارش PDF"""
        if output_path is None:
            output_path = f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # ایجاد نمودارها
        charts = self._generate_charts(data)
        
        # ایجاد PDF
        pdf = self._create_pdf_document(data, charts)
        pdf.output(output_path)
        
        # حذف فایل‌های نمودار موقت
        for chart_path in charts.values():
            if os.path.exists(chart_path):
                os.remove(chart_path)
        
        return output_path
    
    def _create_pdf_document(self, data: Dict, charts: Dict) -> FPDF:
        """ایجاد سند PDF"""
        class PDF(FPDF):
            def header(self):
                if os.path.exists(self.logo_path):
                    self.image(self.logo_path, 10, 8, 25)
                self.set_font('Vazir', 'B', 12)
                self.cell(0, 10, 'گزارش مالی شخصی', 0, 1, 'C')
                self.ln(10)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Vazir', 'I', 8)
                self.cell(0, 10, f'صفحه {self.page_no()}', 0, 0, 'C')
        
        pdf = PDF()
        pdf.add_font('Vazir', '', self.font_path, uni=True)
        pdf.add_font('Vazir', 'B', self.font_path, uni=True)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Vazir', '', 10)
        
        # افزودن خلاصه وضعیت مالی
        self._add_summary_section(pdf, data['insights'])
        
        # افزودن نمودارها
        if 'category_chart' in charts:
            pdf.add_page()
            pdf.set_font('Vazir', 'B', 12)
            pdf.cell(0, 10, 'تحلیل دسته‌بندی هزینه‌ها', 0, 1, 'R')
            pdf.image(charts['category_chart'], w=150)
        
        if 'trend_chart' in charts:
            pdf.add_page()
            pdf.set_font('Vazir', 'B', 12)
            pdf.cell(0, 10, 'روند مالی ماهانه', 0, 1, 'R')
            pdf.image(charts['trend_chart'], w=150)
        
        # افزودن جدول تراکنش‌ها
        pdf.add_page()
        self._add_transactions_table(pdf, data['transactions'])
        
        return pdf
    
    def _add_summary_section(self, pdf: FPDF, insights: Dict):
        """افزودن بخش خلاصه وضعیت مالی"""
        pdf.set_font('Vazir', 'B', 12)
        pdf.cell(0, 10, 'خلاصه وضعیت مالی', 0, 1, 'R')
        pdf.ln(5)
        
        pdf.set_font('Vazir', '', 10)
        summary_items = [
            ('درآمد کل', f"{insights['total_income']:,.0f} $"),
            ('هزینه کل', f"{insights['total_expense']:,.0f} $"),
            ('تراز خالص', f"{insights['net_balance']:,.0f} $"),
            ('نرخ پس‌انداز', f"{insights['savings_rate']:.1%}"),
            ('بیشترین هزینه', insights['top_category']),
            ('تراکنش‌های تکراری', f"{insights['recurring_count']} مورد"),
            ('اشتراک‌های فعال', f"{insights['subscription_count']} مورد"),
            ('هزینه‌های غیرمعمول', f"{insights['unusual_count']} مورد")
        ]
        
        for label, value in summary_items:
            pdf.cell(0, 8, f'{label}: {value}', 0, 1, 'R')
        
        pdf.ln(10)
    
    def _add_transactions_table(self, pdf: FPDF, transactions: pd.DataFrame):
        """افزودن جدول تراکنش‌ها"""
        pdf.set_font('Vazir', 'B', 12)
        pdf.cell(0, 10, 'لیست تراکنش‌ها', 0, 1, 'R')
        pdf.ln(5)
        
        # هدر جدول
        headers = ['تاریخ', 'توضیحات', 'مبلغ', 'دسته‌بندی', 'نوع']
        col_widths = [30, 60, 30, 30, 30]
        
        pdf.set_font('Vazir', 'B', 8)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
        pdf.ln()
        
        # داده‌های جدول
        pdf.set_font('Vazir', '', 8)
        for _, row in transactions.head(20).iterrows():
            pdf.cell(col_widths[0], 10, row['date'].strftime('%Y-%m-%d'), 1, 0, 'C')
            pdf.cell(col_widths[1], 10, str(row['description'])[:20] + '...', 1, 0, 'R')
            pdf.cell(col_widths[2], 10, f"{row['amount']:,.0f}", 1, 0, 'L')
            pdf.cell(col_widths[3], 10, row['category'], 1, 0, 'C')
            pdf.cell(col_widths[4], 10, row['type'], 1, 0, 'C')
            pdf.ln()
    
    def _generate_html_report(self, data: Dict, output_path: str = None, template_name: str = None) -> str:
        """تولید گزارش HTML"""
        if output_path is None:
            output_path = f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        if self.jinja_env is None:
            # استفاده از قالب پیش‌فرض
            html_content = self._create_default_html_template(data)
        else:
            # استفاده از قالب سفارشی
            template_name = template_name or 'default.html'
            template = self.jinja_env.get_template(template_name)
            html_content = template.render(**data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _create_default_html_template(self, data: Dict) -> str:
        """ایجاد قالب HTML پیش‌فرض"""
        # ایجاد نمودارها به صورت base64
        charts = self._generate_charts(data, return_base64=True)
        
        html = f"""
        <!DOCTYPE html>
        <html dir="rtl">
        <head>
            <meta charset="UTF-8">
            <title>گزارش مالی</title>
            <style>
                body {{ font-family: Vazir, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
                .card {{ background: #f5f5f5; padding: 15px; border-radius: 8px; }}
                .card h3 {{ margin: 0 0 10px 0; }}
                .card .value {{ font-size: 24px; font-weight: bold; }}
                .chart-container {{ margin: 20px 0; text-align: center; }}
                .chart-container img {{ max-width: 100%; height: auto; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>گزارش مالی شخصی</h1>
                <p>تاریخ تولید: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <div class="card">
                    <h3>درآمد کل</h3>
                    <div class="value">{data['insights']['total_income']:,.0f} $</div>
                </div>
                <div class="card">
                    <h3>هزینه کل</h3>
                    <div class="value">{data['insights']['total_expense']:,.0f} $</div>
                </div>
                <div class="card">
                    <h3>تراز خالص</h3>
                    <div class="value">{data['insights']['net_balance']:,.0f} $</div>
                </div>
                <div class="card">
                    <h3>نرخ پس‌انداز</h3>
                    <div class="value">{data['insights']['savings_rate']:.1%}</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>تحلیل دسته‌بندی هزینه‌ها</h2>
                <img src="data:image/png;base64,{charts.get('category_chart', '')}" alt="Category Chart">
            </div>
            
            <div class="chart-container">
                <h2>روند مالی ماهانه</h2>
                <img src="data:image/png;base64,{charts.get('trend_chart', '')}" alt="Trend Chart">
            </div>
            
            <h2>لیست تراکنش‌ها</h2>
            <table>
                <tr>
                    <th>تاریخ</th>
                    <th>توضیحات</th>
                    <th>مبلغ</th>
                    <th>دسته‌بندی</th>
                    <th>نوع</th>
                </tr>
        """
        
        for _, row in data['transactions'].head(20).iterrows():
            html += f"""
                <tr>
                    <td>{row['date'].strftime('%Y-%m-%d')}</td>
                    <td>{row['description']}</td>
                    <td>{row['amount']:,.0f}</td>
                    <td>{row['category']}</td>
                    <td>{row['type']}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
    
    def _generate_excel_report(self, data: Dict, output_path: str = None) -> str:
        """تولید گزارش Excel"""
        if output_path is None:
            output_path = f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # شیت خلاصه
            summary_data = {
                'معیار': ['درآمد کل', 'هزینه کل', 'تراز خالص', 'نرخ پس‌انداز', 'بیشترین هزینه', 
                         'تراکنش‌های تکراری', 'اشتراک‌های فعال', 'هزینه‌های غیرمعمول'],
                'مقدار': [
                    data['insights']['total_income'],
                    data['insights']['total_expense'],
                    data['insights']['net_balance'],
                    data['insights']['savings_rate'],
                    data['insights']['top_category'],
                    data['insights']['recurring_count'],
                    data['insights']['subscription_count'],
                    data['insights']['unusual_count']
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='خلاصه', index=False)
            
            # شیت تراکنش‌ها
            transactions_df = data['transactions'].copy()
            transactions_df.to_excel(writer, sheet_name='تراکنش‌ها', index=False)
            
            # شیت تحلیل دسته‌بندی
            if 'category_analysis' in data:
                category_df = data['category_analysis']
                category_df.to_excel(writer, sheet_name='تحلیل دسته‌بندی', index=False)
            
            # شیت روند ماهانه
            if 'monthly_trend' in data:
                trend_df = data['monthly_trend']
                trend_df.to_excel(writer, sheet_name='روند ماهانه', index=False)
            
            # فرمت‌بندی شیت‌ها
            workbook = writer.book
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                
                # تنظیم عرض ستون‌ها
                if sheet_name == 'تراکنش‌ها':
                    worksheet.set_column('A:A', 12)  # تاریخ
                    worksheet.set_column('B:B', 40)  # توضیحات
                    worksheet.set_column('C:C', 15)  # مبلغ
                    worksheet.set_column('D:D', 15)  # دسته‌بندی
                    worksheet.set_column('E:E', 10)  # نوع
                
                # افزودن فیلتر
                worksheet.autofilter(0, 0, 0, len(transactions_df.columns) - 1)
        
        return output_path
    
    def _generate_charts(self, data: Dict, return_base64: bool = False) -> Dict:
        """تولید نمودارهای گزارش"""
        charts = {}
        
        # نمودار دسته‌بندی هزینه‌ها
        if 'transactions' in data:
            expense_data = data['transactions'][data['transactions']['type'] == 'هزینه']
            if not expense_data.empty:
                plt.figure(figsize=(8, 6))
                category_data = expense_data.groupby('category')['amount_usd'].sum().reset_index()
                
                # مرتب‌سازی برای نمایش بهتر
                category_data = category_data.sort_values('amount_usd', ascending=False)
                
                # ایجاد نمودار پای
                plt.pie(category_data['amount_usd'], labels=category_data['category'], autopct='%1.1f%%')
                plt.title('تحلیل دسته‌بندی هزینه‌ها', fontname='Vazir')
                plt.axis('equal')
                
                if return_base64:
                    charts['category_chart'] = self._fig_to_base64(plt.gcf())
                else:
                    chart_path = 'category_chart.png'
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    charts['category_chart'] = chart_path
                plt.close()
        
        # نمودار روند مالی
        if 'monthly_trend' in data:
            plt.figure(figsize=(10, 6))
            trend_data = data['monthly_trend']
            
            # ایجاد نمودار خطی
            plt.plot(trend_data['date'], trend_data['income'], label='درآمد', marker='o')
            plt.plot(trend_data['date'], trend_data['expense'], label='هزینه', marker='o')
            plt.plot(trend_data['date'], trend_data['net'], label='تراز خالص', marker='o')
            
            plt.title('روند مالی ماهانه', fontname='Vazir')
            plt.xlabel('تاریخ', fontname='Vazir')
            plt.ylabel('مبلغ (دلار)', fontname='Vazir')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            
            if return_base64:
                charts['trend_chart'] = self._fig_to_base64(plt.gcf())
            else:
                chart_path = 'trend_chart.png'
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                charts['trend_chart'] = chart_path
            plt.close()
        
        return charts
    
    def _fig_to_base64(self, fig) -> str:
        """تبدیل نمودار به base64"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return image_base64
    
    def _format_persian_text(self, text: str) -> str:
        """قالب‌بندی متن فارسی برای PDF"""
        if not PERSIAN_SUPPORT:
            return text
        
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
    
    def _format_number(self, number: Union[int, float], currency: str = None) -> str:
        """قالب‌بندی اعداد"""
        currency = currency or self.default_currency
        
        if currency == 'IRR':
            return f"{number:,.0f} ریال"
        else:
            return f"{number:,.2f} {currency}"
    
    def _format_date(self, date: datetime, format_str: str = None) -> str:
        """قالب‌بندی تاریخ"""
        format_str = format_str or self.date_format
        return date.strftime(format_str)

class AdvancedReportGenerator(FinancialReportGenerator):
    """
    نسخه پیشرفته تولیدکننده گزارش‌ها با قابلیت‌های اضافی
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.custom_styles = self.config.get('custom_styles', {})
        self.include_recommendations = self.config.get('include_recommendations', True)
    
    def generate_comprehensive_report(self, data: Dict, output_path: str = None) -> str:
        """تولید گزارش جامع با تحلیل‌های پیشرفته"""
        # افزودن تحلیل‌های پیشرفته به داده‌ها
        enhanced_data = self._enhance_report_data(data)
        
        # تولید گزارش PDF با تمام بخش‌ها
        return self._generate_pdf_report(enhanced_data, output_path)
    
    def _enhance_report_data(self, data: Dict) -> Dict:
        """افزودن تحلیل‌های پیشرفته به داده‌های گزارش"""
        enhanced = data.copy()
        
        # افزودن پیشنهادات مالی
        if self.include_recommendations:
            enhanced['recommendations'] = self._generate_recommendations(data)
        
        # افزودن تحلیل‌های پیشرفته
        enhanced['advanced_analysis'] = self._generate_advanced_analysis(data)
        
        # افزودن اهداف مالی
        enhanced['financial_goals'] = self._generate_financial_goals(data)
        
        return enhanced
    
    def _generate_recommendations(self, data: Dict) -> List[str]:
        """تولید پیشنهادات مالی هوشمند"""
        recommendations = []
        insights = data['insights']
        
        # پیشنهاد برای نرخ پس‌انداز
        if insights['savings_rate'] < 0.1:
            recommendations.append(
                "نرخ پس‌انداز شما کمتر از 10% است. سعی کنید هزینه‌های غیرضروری را کاهش دهید."
            )
        
        # پیشنهاد برای هزینه‌های غیرمعمول
        if insights['unusual_count'] > 0:
            recommendations.append(
                f"{insights['unusual_count']} تراکنش غیرمعمول شناسایی شد. این موارد را بررسی کنید."
            )
        
        # پیشنهاد برای اشتراک‌ها
        if insights['subscription_count'] > 5:
            recommendations.append(
                f"شما {insights['subscription_count']} اشتراک فعال دارید. بررسی کنید که همه آن‌ها لازم هستند."
            )
        
        # پیشنهاد برای سرمایه‌گذاری
        if insights['savings_rate'] > 0.2:
            recommendations.append(
                "نرخ پس‌انداز شما خوب است. بخشی از آن را در سرمایه‌گذاری‌های مطمئن قرار دهید."
            )
        
        return recommendations
    
    def _generate_advanced_analysis(self, data: Dict) -> Dict:
        """تولید تحلیل‌های پیشرفته"""
        analysis = {}
        transactions = data['transactions']
        
        # تحلیل فصلی
        if 'quarter' in transactions.columns:
            quarterly_analysis = transactions.groupby('quarter').agg({
                'amount_usd': ['sum', 'count'],
                'category': lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'
            }).round(2)
            analysis['quarterly'] = quarterly_analysis.to_dict()
        
        # تحلیل روزهای هفته
        if 'weekday' in transactions.columns:
            weekday_analysis = transactions.groupby('weekday')['amount_usd'].mean().sort_values(ascending=False)
            analysis['weekday_spending'] = weekday_analysis.to_dict()
        
        # تحلیل روندها
        if 'monthly_trend' in data:
            trend_data = data['monthly_trend']
            analysis['trends'] = {
                'income_growth': self._calculate_growth_rate(trend_data['income']),
                'expense_growth': self._calculate_growth_rate(trend_data['expense']),
                'savings_trend': self._calculate_growth_rate(trend_data['net'])
            }
        
        return analysis
    
    def _calculate_growth_rate(self, series: pd.Series) -> float:
        """محاسبه نرخ رشد"""
        if len(series) < 2:
            return 0.0
        
        first = series.iloc[0]
        last = series.iloc[-1]
        
        if first == 0:
            return 0.0
        
        return ((last - first) / first) * 100
    
    def _generate_financial_goals(self, data: Dict) -> Dict:
        """تولید اهداف مالی پیشنهادی"""
        goals = {}
        insights = data['insights']
        
        # هدف پس‌انداز
        current_savings_rate = insights['savings_rate']
        if current_savings_rate < 0.2:
            goals['savings_target'] = {
                'current': f"{current_savings_rate:.1%}",
                'target': "20%",
                'description': "افزایش نرخ پس‌انداز به 20%"
            }
        
        # هدف کاهش هزینه‌ها
        if insights['total_expense'] > 0:
            reduction_target = insights['total_expense'] * 0.1
            goals['expense_reduction'] = {
                'current': f"{insights['total_expense']:,.0f} $",
                'target': f"{insights['total_expense'] - reduction_target:,.0f} $",
                'description': "کاهش 10% هزینه‌ها"
            }
        
        # هدف سرمایه‌گذاری
        if insights['total_income'] > 0:
            investment_target = insights['total_income'] * 0.15
            goals['investment'] = {
                'current': "0 $",
                'target': f"{investment_target:,.0f} $",
                'description': "سرمایه‌گذاری 15% از درآمد"
            }
        
        return goals
