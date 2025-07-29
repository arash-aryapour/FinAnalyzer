# core/ml_engine.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class TransactionCategorizer:
    """
    موتور یادگیری ماشین برای دسته‌بندی هوشمند تراکنش‌های مالی
    شامل روش‌های مبتنی بر قوانین، یادگیری ماشین و مدل‌های پیشرفته NLP
    """
    
    def __init__(self, config: Dict = None):
        """
        مقداردهی اولیه موتور دسته‌بندی
        
        Args:
            config (Dict): تنظیمات سفارشی برای مدل
        """
        self.config = config or {}
        self.categories = [
            "درآمد", "خرید", "قبوض", "حمل و نقل", "غذا",
            "سرمایه گذاری", "سلامت", "آموزش", "تفریح", "سایر"
        ]
        
        # تنظیمات پیش‌فرض
        self.model_type = self.config.get('model_type', 'hybrid')  # rule, ml, hybrid, zero_shot
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.model_path = self.config.get('model_path', 'models/transaction_classifier.pkl')
        
        # بارگذاری مدل‌ها
        self._load_models()
    
    def _load_models(self):
        """بارگذاری مدل‌های یادگیری ماشین"""
        # مدل zero-shot classification
        try:
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Error loading zero-shot model: {e}")
            self.zero_shot_classifier = None
        
        # مدل یادگیری ماشین سفارشی
        if os.path.exists(self.model_path):
            try:
                self.custom_classifier = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.model_path.replace('.pkl', '_vectorizer.pkl'))
            except Exception as e:
                print(f"Error loading custom model: {e}")
                self.custom_classifier = None
                self.vectorizer = None
        else:
            self.custom_classifier = None
            self.vectorizer = None
    
    def categorize_transactions(self, descriptions: List[str], method: str = None) -> List[str]:
        """
        دسته‌بندی لیستی از تراکنش‌ها
        
        Args:
            descriptions (List[str]): لیست توضیحات تراکنش‌ها
            method (str): روش دسته‌بندی (rule, ml, hybrid, zero_shot)
            
        Returns:
            List[str]: لیست دسته‌بندی‌های پیش‌بینی شده
        """
        method = method or self.model_type
        categories = []
        
        if method == 'rule':
            categories = self._rule_based_categorize(descriptions)
        elif method == 'ml':
            categories = self._ml_based_categorize(descriptions)
        elif method == 'zero_shot':
            categories = self._zero_shot_categorize(descriptions)
        elif method == 'hybrid':
            categories = self._hybrid_categorize(descriptions)
        else:
            raise ValueError(f"Unknown categorization method: {method}")
        
        return categories
    
    def _rule_based_categorize(self, descriptions: List[str]) -> List[str]:
        """دسته‌بندی مبتنی بر قوانین"""
        rules = {
            "درآمد": [
                r'حقوق', r'سود', r'فروش', r'درآمد', r'واریز', r'deposit', 
                r'income', r'salary', r'bonus', r'refund'
            ],
            "خرید": [
                r'خرید', r'دیجی‌کالا', r'فروشگاه', r'shop', r'purchase',
                r'amazon', r'ebay', r'aliexpress'
            ],
            "قبوض": [
                r'قبض', r'برق', r'آب', r'گاز', r'bill', r'utility',
                r'mobile', r'اینترنت', r'telecom'
            ],
            "حمل و نقل": [
                r'بنزین', r'تاکسی', r'مترو', r'اتوبوس', r'gas', r'taxi',
                r'uber', r'snapp', r'پارکینگ'
            ],
            "غذا": [
                r'سوپرمارکت', r'رستوران', r'غذا', r'restaurant', r'food',
                r'کافه', r'پیتزا', r'ساندویچ'
            ],
            "سرمایه گذاری": [
                r'بورس', r'سهام', r'سرمایه', r'stock', r'investment',
                r'ارز دیجیتال', r'طلا', r'savings'
            ],
            "سلامت": [
                r'درمان', r'دارو', r'بیمارستان', r'health', r'medical',
                r'دندانپزشکی', r'داروخانه'
            ],
            "آموزش": [
                r'کتاب', r'دوره', r'آموزش', r'book', r'course',
                r'دانشگاه', r'مدرسه', r'workshop'
            ],
            "تفریح": [
                r'سینما', r'تفریح', r'تئاتر', r'cinema', r'entertainment',
                r'کنسرت', r'ورزش', r'گیم'
            ]
        }
        
        categories = []
        for desc in descriptions:
            if pd.isna(desc) or desc == "":
                categories.append("سایر")
                continue
                
            desc_lower = str(desc).lower()
            matched_category = "سایر"
            
            for category, patterns in rules.items():
                for pattern in patterns:
                    if re.search(pattern, desc_lower, re.IGNORECASE):
                        matched_category = category
                        break
                if matched_category != "سایر":
                    break
            
            categories.append(matched_category)
        
        return categories
    
    def _ml_based_categorize(self, descriptions: List[str]) -> List[str]:
        """دسته‌بندی مبتنی بر مدل یادگیری ماشین سفارشی"""
        if self.custom_classifier is None or self.vectorizer is None:
            print("Custom ML model not available. Falling back to rule-based.")
            return self._rule_based_categorize(descriptions)
        
        # پیش‌پردازش متن
        processed_desc = [self._preprocess_text(desc) for desc in descriptions]
        
        # تبدیل متن به بردار
        X = self.vectorizer.transform(processed_desc)
        
        # پیش‌بینی دسته‌ها
        predictions = self.custom_classifier.predict(X)
        probabilities = self.custom_classifier.predict_proba(X)
        
        # بررسی آستانه اطمینان
        categories = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            max_prob = np.max(probs)
            if max_prob >= self.confidence_threshold:
                categories.append(pred)
            else:
                # اگر اطمینان کافی نبود، استفاده از روش مبتنی بر قوانین
                categories.append(self._rule_based_categorize([descriptions[i]])[0])
        
        return categories
    
    def _zero_shot_categorize(self, descriptions: List[str]) -> List[str]:
        """دسته‌بندی با استفاده از مدل zero-shot classification"""
        if self.zero_shot_classifier is None:
            print("Zero-shot model not available. Falling back to rule-based.")
            return self._rule_based_categorize(descriptions)
        
        categories = []
        batch_size = 8  # پردازش دسته‌ای برای بهبود عملکرد
        
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i:i+batch_size]
            batch = [str(desc) if pd.notna(desc) else "" for desc in batch]
            
            try:
                results = self.zero_shot_classifier(
                    batch,
                    candidate_labels=self.categories,
                    hypothesis_template="این تراکنش مربوط به {} است."
                )
                
                for result in results:
                    if result['scores'][0] >= self.confidence_threshold:
                        categories.append(result['labels'][0])
                    else:
                        # اگر اطمینان کافی نبود، استفاده از روش مبتنی بر قوانین
                        rule_cat = self._rule_based_categorize([result['sequence']])[0]
                        categories.append(rule_cat)
            except Exception as e:
                print(f"Error in zero-shot classification: {e}")
                # در صورت خطا، استفاده از روش مبتنی بر قوانین
                rule_cats = self._rule_based_categorize(batch)
                categories.extend(rule_cats)
        
        return categories
    
    def _hybrid_categorize(self, descriptions: List[str]) -> List[str]:
        """دسته‌بندی ترکیبی از چندین روش"""
        categories = []
        
        for desc in descriptions:
            # ابتدا استفاده از روش مبتنی بر قوانین
            rule_cat = self._rule_based_categorize([desc])[0]
            
            # اگر دسته‌بندی قطعی بود، استفاده از آن
            if rule_cat != "سایر":
                categories.append(rule_cat)
                continue
            
            # در غیر این صورت، استفاده از مدل zero-shot
            if self.zero_shot_classifier is not None:
                try:
                    result = self.zero_shot_classifier(
                        str(desc) if pd.notna(desc) else "",
                        candidate_labels=self.categories,
                        hypothesis_template="این تراکنش مربوط به {} است."
                    )
                    
                    if result['scores'][0] >= self.confidence_threshold:
                        categories.append(result['labels'][0])
                    else:
                        categories.append(rule_cat)
                except:
                    categories.append(rule_cat)
            else:
                categories.append(rule_cat)
        
        return categories
    
    def _preprocess_text(self, text: str) -> str:
        """پیش‌پردازش متن برای مدل یادگیری ماشین"""
        if pd.isna(text):
            return ""
        
        # تبدیل به حروف کوچک
        text = str(text).lower()
        
        # حذف اعداد و کاراکترهای خاص
        text = re.sub(r'[^a-zA-Z0-9\s\u0600-\u06FF]', ' ', text)
        
        # حذف فضاهای اضافی
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train_custom_model(self, df: pd.DataFrame, text_column: str = 'description', 
                          label_column: str = 'category', test_size: float = 0.2):
        """
        آموزش مدل یادگیری ماشین سفارشی
        
        Args:
            df (pd.DataFrame): دیتافریم آموزش
            text_column (str): نام ستون متن
            label_column (str): نام ستون برچسب
            test_size (float): نسبت داده‌های تست
        """
        # پیش‌پردازش داده‌ها
        df = df.dropna(subset=[text_column, label_column])
        df[text_column] = df[text_column].apply(self._preprocess_text)
        
        # تقسیم داده‌ها
        X_train, X_test, y_train, y_test = train_test_split(
            df[text_column], 
            df[label_column], 
            test_size=test_size, 
            random_state=42
        )
        
        # ایجاد بردار TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # آموزش مدل
        self.custom_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        self.custom_classifier.fit(X_train_vec, y_train)
        
        # ارزیابی مدل
        y_pred = self.custom_classifier.predict(X_test_vec)
        print("Model Evaluation:")
        print(classification_report(y_test, y_pred))
        
        # ذخیره مدل
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.custom_classifier, self.model_path)
        joblib.dump(self.vectorizer, self.model_path.replace('.pkl', '_vectorizer.pkl'))
        
        print(f"Model saved to {self.model_path}")
    
    def analyze_category_patterns(self, df: pd.DataFrame) -> Dict:
        """
        تحلیل الگوهای دسته‌بندی‌ها
        
        Args:
            df (pd.DataFrame): دیتافریم تراکنش‌ها
            
        Returns:
            Dict: تحلیل الگوها
        """
        analysis = {}
        
        # توزیع دسته‌بندی‌ها
        category_dist = df['category'].value_counts().to_dict()
        analysis['category_distribution'] = category_dist
        
        # میانگین مبلغ هر دسته
        category_avg = df.groupby('category')['amount_usd'].mean().to_dict()
        analysis['category_average_amount'] = category_avg
        
        # کلمات کلیدی هر دسته
        category_keywords = {}
        for category in df['category'].unique():
            texts = df[df['category'] == category]['description'].dropna()
            if len(texts) > 0:
                # استخراج کلمات کلیدی با TF-IDF
                vectorizer = TfidfVectorizer(max_features=10, ngram_range=(1, 2))
                try:
                    X = vectorizer.fit_transform(texts)
                    feature_names = vectorizer.get_feature_names_out()
                    scores = X.sum(axis=0).A1
                    top_keywords = [feature_names[i] for i in scores.argsort()[-5:]]
                    category_keywords[category] = top_keywords
                except:
                    category_keywords[category] = []
        
        analysis['category_keywords'] = category_keywords
        
        return analysis
    
    def suggest_category_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        پیشنهاد تصحیح دسته‌بندی‌های احتمالاً اشتباه
        
        Args:
            df (pd.DataFrame): دیتافریم تراکنش‌ها
            
        Returns:
            pd.DataFrame: تراکنش‌های با دسته‌بندی مشکوک
        """
        suspicious = df.copy()
        suspicious['suggested_category'] = None
        suspicious['confidence'] = None
        
        for idx, row in suspicious.iterrows():
            desc = row['description']
            current_cat = row['category']
            
            if pd.isna(desc):
                continue
            
            # پیش‌بینی با مدل zero-shot
            if self.zero_shot_classifier is not None:
                try:
                    result = self.zero_shot_classifier(
                        str(desc),
                        candidate_labels=self.categories,
                        hypothesis_template="این تراکنش مربوط به {} است."
                    )
                    
                    top_cat = result['labels'][0]
                    confidence = result['scores'][0]
                    
                    # اگر دسته‌بندی پیشنهادی با دسته فعلی متفاوت بود و اطمینان بالا
                    if top_cat != current_cat and confidence > 0.8:
                        suspicious.at[idx, 'suggested_category'] = top_cat
                        suspicious.at[idx, 'confidence'] = confidence
                except:
                    pass
        
        # فقط تراکنش‌های با پیشنهاد تصحیح
        return suspicious[suspicious['suggested_category'].notna()]
    
    def get_category_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        محاسبه آمار دقیق هر دسته‌بندی
        
        Args:
            df (pd.DataFrame): دیتافریم تراکنش‌ها
            
        Returns:
            pd.DataFrame: آمار دسته‌بندی‌ها
        """
        stats = df.groupby('category').agg(
            count=('amount_usd', 'count'),
            total_amount=('amount_usd', 'sum'),
            avg_amount=('amount_usd', 'mean'),
            min_amount=('amount_usd', 'min'),
            max_amount=('amount_usd', 'max'),
            std_amount=('amount_usd', 'std')
        ).reset_index()
        
        # محاسبه درصد
        total_count = len(df)
        stats['percentage'] = (stats['count'] / total_count * 100).round(2)
        
        return stats.sort_values('count', ascending=False)
