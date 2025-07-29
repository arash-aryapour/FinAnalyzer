import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# افزودن مسیر ماژول‌ها
sys.path.append('./core')
sys.path.append('./utils')

# وارد کردن ماژول‌های مورد نیاز
from data_processor import FinancialDataProcessor
from ml_engine import TransactionCategorizer
from report_generator import generate_financial_report

# تنظیمات صفحه
st.set_page_config(
    page_title="FinAnalyzer - تحلیلگر مالی هوشمند",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تابع برای بارگذاری داده‌ها
def load_data():
    uploaded_file = st.sidebar.file_uploader(
        "فایل مالی خود را آپلود کنید",
        type=['csv'],
        help="فایل CSV با ستون‌های Date, Description, Amount, Category, Currency"
    )
    
    if uploaded_file:
        return uploaded_file
    else:
        # استفاده از داده‌های نمونه
        sample_path = "data/sample_data.csv"
        if os.path.exists(sample_path):
            return sample_path
        return None

# تابع برای پردازش داده‌ها با کش کردن
@st.cache_data
def process_data(file_path):
    processor = FinancialDataProcessor()
    df = processor.load_data(file_path)
    df = processor.preprocess_data(df)
    df = processor.detect_patterns(df)
    insights = processor.generate_insights(df)
    return df, insights

# تابع برای نمایش خلاصه وضعیت مالی
def show_summary_cards(insights):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="درآمد کل",
            value=f"{insights['total_income']:,.0f} $",
            delta="12%"
        )
    
    with col2:
        st.metric(
            label="هزینه کل",
            value=f"{insights['total_expense']:,.0f} $",
            delta="-8%"
        )
    
    with col3:
        st.metric(
            label="تراز خالص",
            value=f"{insights['net_balance']:,.0f} $",
            delta="5%"
        )
    
    with col4:
        st.metric(
            label="نرخ پس‌انداز",
            value=f"{insights['savings_rate']:.1%}",
            delta="2.1%"
        )

# تابع برای نمایش نمودارها
def show_charts(df):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 روند مالی ماهانه")
        monthly_data = df.groupby(['year', 'month'])['amount_usd'].sum().reset_index()
        monthly_data['date'] = pd.to_datetime(
            monthly_data['year'].astype(str) + '-' + monthly_data['month'].astype(str) + '-01'
        )
        
        fig = px.line(
            monthly_data,
            x='date',
            y='amount_usd',
            title='روند درآمد و هزینه',
            labels={'amount_usd': 'مبلغ (دلار)', 'date': 'تاریخ'}
        )
        fig.add_hline(y=0, line_dash="dot", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🥧 تحلیل دسته‌بندی هزینه‌ها")
        expense_data = df[df['type'] == 'هزینه']
        category_data = expense_data.groupby('category')['amount_usd'].sum().reset_index()
        
        fig = px.pie(
            category_data,
            values='amount_usd',
            names='category',
            title='توزیع هزینه‌ها'
        )
        st.plotly_chart(fig, use_container_width=True)

# تابع برای نمایش جدول تراکنش‌ها
def show_transactions_table(df):
    st.subheader("📋 لیست تراکنش‌ها")
    
    # فیلترها
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_filter = st.date_input(
            "فیلتر تاریخ",
            value=(datetime.now() - timedelta(days=30), datetime.now())
        )
    
    with col2:
        category_filter = st.multiselect(
            "فیلتر دسته‌بندی",
            options=df['category'].unique(),
            default=df['category'].unique()
        )
    
    with col3:
        type_filter = st.multiselect(
            "فیلتر نوع",
            options=df['type'].unique(),
            default=df['type'].unique()
        )
    
    # اعمال فیلترها
    filtered_data = df.copy()
    filtered_data = filtered_data[
        (filtered_data['date'] >= pd.to_datetime(date_filter[0])) &
        (filtered_data['date'] <= pd.to_datetime(date_filter[1])) &
        (filtered_data['category'].isin(category_filter)) &
        (filtered_data['type'].isin(type_filter))
    ]
    
    st.dataframe(
        filtered_data.sort_values('date', ascending=False),
        use_container_width=True,
        column_config={
            "date": "تاریخ",
            "description": "توضیحات",
            "amount": "مبلغ",
            "category": "دسته‌بندی",
            "type": "نوع",
            "is_recurring": "تکراری",
            "is_subscription": "اشتراک",
            "is_unusual": "غیرمعمول"
        }
    )

# تابع برای تولید گزارش
def generate_report(insights, df):
    if st.button("📥 تولید گزارش PDF"):
        with st.spinner("در حال تولید گزارش..."):
            output_path = "financial_report.pdf"
            generate_financial_report(insights, df, output_path)
            
            with open(output_path, "rb") as f:
                st.download_button(
                    label="دانلود گزارش",
                    data=f,
                    file_name="financial_report.pdf",
                    mime="application/pdf"
                )
            
            os.remove(output_path)

# تابع اصلی برنامه
def main():
    # عنوان اصلی
    st.title("💰 ابزار تحلیل مالی هوشمند")
    st.markdown("---")
    
    # بارگذاری داده‌ها
    file_path = load_data()
    
    if file_path:
        # پردازش داده‌ها
        df, insights = process_data(file_path)
        
        # نمایش خلاصه وضعیت مالی
        show_summary_cards(insights)
        
        # نمایش نمودارها
        show_charts(df)
        
        # نمایش جدول تراکنش‌ها
        show_transactions_table(df)
        
        # تولید گزارش
        generate_report(insights, df)
        
        # نمایش بینش‌های کلیدی
        st.subheader("💡 بینش‌های کلیدی")
        st.json(insights)
    else:
        st.warning("لطفاً یک فایل CSV آپلود کنید یا از داده‌های نمونه استفاده کنید")

# اجرای برنامه
if __name__ == "__main__":
    main()
