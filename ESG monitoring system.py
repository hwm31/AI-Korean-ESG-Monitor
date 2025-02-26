# ESG 모니터링 시스템 구현 코드

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup
import tweepy
import pdfplumber
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import pickle
import schedule
import time
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("esg_monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# NLTK 리소스 다운로드
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ESG 관련 키워드 정의
esg_keywords = {
    'environmental': [
        'carbon emissions', 'climate change', 'renewable energy', 'sustainability', 
        'pollution', 'waste management', 'recycling', 'water conservation', 
        'deforestation', 'biodiversity', 'green energy', 'carbon footprint', 
        'environmental impact', 'clean energy', 'greenhouse gas'
    ],
    'social': [
        'human rights', 'diversity', 'inclusion', 'gender equality', 'labor rights', 
        'community engagement', 'employee welfare', 'health and safety', 'fair trade', 
        'social responsibility', 'supply chain ethics', 'work-life balance', 
        'social impact', 'ethical sourcing', 'stakeholder engagement'
    ],
    'governance': [
        'board diversity', 'executive compensation', 'shareholder rights', 
        'business ethics', 'transparency', 'corruption', 'compliance', 
        'risk management', 'audit committee', 'corporate governance', 
        'whistleblower protection', 'accounting practices', 'data privacy', 
        'stakeholder governance', 'regulatory compliance'
    ]
}

# ESG 카테고리별 가중치 설정 (예시)
esg_weights = {
    'environmental': 0.35,
    'social': 0.35,
    'governance': 0.30
}

# 데이터 수집 클래스
class DataCollector:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        
        # Twitter API 인증 설정
        auth = tweepy.OAuthHandler(
            api_keys['twitter']['consumer_key'], 
            api_keys['twitter']['consumer_secret']
        )
        auth.set_access_token(
            api_keys['twitter']['access_token'], 
            api_keys['twitter']['access_token_secret']
        )
        self.twitter_api = tweepy.API(auth)
        
    def collect_news_data(self, company_name, days_back=7):
        """뉴스 기사 데이터 수집"""
        logger.info(f"뉴스 데이터 수집 시작: {company_name}")
        collected_data = []
        
        # 검색어 설정
        search_query = f"{company_name} (ESG OR Environmental OR Social OR Governance)"
        
        # 날짜 범위 설정
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_range = f"{start_date.strftime('%Y-%m-%d')} TO {end_date.strftime('%Y-%m-%d')}"
        
        # 뉴스 API 요청 (예시: NewsAPI)
        try:
            response = requests.get(
                f"https://newsapi.org/v2/everything",
                params={
                    "q": search_query,
                    "from": start_date.strftime('%Y-%m-%d'),
                    "to": end_date.strftime('%Y-%m-%d'),
                    "language": "en",
                    "sortBy": "relevancy",
                    "apiKey": self.api_keys['news_api']
                }
            )
            
            if response.status_code == 200:
                news_data = response.json()
                
                for article in news_data.get('articles', []):
                    collected_data.append({
                        'source': 'news',
                        'source_name': article.get('source', {}).get('name', 'Unknown'),
                        'date': article.get('publishedAt', ''),
                        'title': article.get('title', ''),
                        'content': article.get('description', '') + ' ' + article.get('content', ''),
                        'url': article.get('url', '')
                    })
                    
                logger.info(f"{len(collected_data)} 개의 뉴스 기사 수집 완료")
            else:
                logger.error(f"뉴스 API 오류: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"뉴스 데이터 수집 중 오류 발생: {str(e)}")
        
        return collected_data
    
    def collect_social_media_data(self, company_name, days_back=7):
        """소셜 미디어 데이터 수집 (Twitter 예시)"""
        logger.info(f"소셜 미디어 데이터 수집 시작: {company_name}")
        collected_data = []
        
        try:
            # 검색어 설정
            search_query = f"{company_name} (ESG OR Environmental OR Social OR Governance) -filter:retweets"
            
            # 날짜 범위 설정
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Twitter API를 통한 데이터 수집
            tweets = tweepy.Cursor(
                self.twitter_api.search_tweets,
                q=search_query,
                lang="en",
                since_id=start_date.strftime('%Y-%m-%d'),
                tweet_mode='extended'
            ).items(100)  # 최대 100개 트윗 수집
            
            for tweet in tweets:
                collected_data.append({
                    'source': 'twitter',
                    'source_name': tweet.user.screen_name,
                    'date': tweet.created_at.strftime('%Y-%m-%d'),
                    'title': '',
                    'content': tweet.full_text,
                    'url': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}"
                })
            
            logger.info(f"{len(collected_data)} 개의 소셜 미디어 데이터 수집 완료")
            
        except Exception as e:
            logger.error(f"소셜 미디어 데이터 수집 중 오류 발생: {str(e)}")
            
        return collected_data
    
    def collect_company_reports(self, company_name, report_urls=None):
        """기업 보고서 수집 및 처리"""
        logger.info(f"기업 보고서 데이터 수집 시작: {company_name}")
        collected_data = []
        
        if not report_urls:
            # 실제 구현에서는 여기에 기업 웹사이트에서 보고서 URL을 찾는 코드를 추가할 수 있음
            logger.warning("보고서 URL이 제공되지 않았습니다.")
            return collected_data
        
        for url in report_urls:
            try:
                # 파일 확장자 확인
                if url.endswith('.pdf'):
                    # PDF 파일 다운로드
                    response = requests.get(url)
                    temp_file = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
                    
                    with open(temp_file, 'wb') as f:
                        f.write(response.content)
                    
                    # PDF 내용 추출
                    text_content = ""
                    with pdfplumber.open(temp_file) as pdf:
                        for page in pdf.pages:
                            text_content += page.extract_text() + " "
                    
                    # 임시 파일 삭제
                    os.remove(temp_file)
                    
                    # 데이터 저장
                    filename = url.split('/')[-1]
                    collected_data.append({
                        'source': 'company_report',
                        'source_name': filename,
                        'date': datetime.now().strftime('%Y-%m-%d'),  # 정확한 발행일이 필요한 경우 메타데이터에서 추출
                        'title': filename,
                        'content': text_content,
                        'url': url
                    })
                    
                    logger.info(f"보고서 처리 완료: {filename}")
                
                else:
                    logger.warning(f"지원되지 않는 파일 형식: {url}")
            
            except Exception as e:
                logger.error(f"보고서 처리 중 오류 발생: {str(e)}")
        
        return collected_data
        
    def collect_all_data(self, company_name, days_back=7, report_urls=None):
        """모든 소스에서 데이터 수집"""
        all_data = []
        
        # 뉴스 데이터 수집
        news_data = self.collect_news_data(company_name, days_back)
        all_data.extend(news_data)
        
        # 소셜 미디어 데이터 수집
        social_data = self.collect_social_media_data(company_name, days_back)
        all_data.extend(social_data)
        
        # 기업 보고서 데이터 수집
        if report_urls:
            report_data = self.collect_company_reports(company_name, report_urls)
            all_data.extend(report_data)
        
        # 데이터프레임 생성
        if all_data:
            df = pd.DataFrame(all_data)
            return df
        else:
            logger.warning("수집된 데이터가 없습니다.")
            return pd.DataFrame()

# 데이터 전처리 클래스
class DataPreprocessor:
    def __init__(self):
        # 불용어 설정
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """텍스트 정제"""
        if not isinstance(text, str):
            return ""
        
        # 소문자 변환
        text = text.lower()
        
        # URL 제거
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # 특수문자 제거
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # 여러 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def tokenize_text(self, text):
        """텍스트 토큰화"""
        # 텍스트 정제
        cleaned_text = self.clean_text(text)
        
        # 토큰화
        tokens = word_tokenize(cleaned_text)
        
        # 불용어 제거
        filtered_tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return filtered_tokens
        
    def preprocess_data(self, df):
        """데이터프레임 전처리"""
        if df.empty:
            logger.warning("전처리할 데이터가 없습니다.")
            return df
        
        # 콘텐츠가 없는 행 제거
        df = df[df['content'].notna()]
        
        # 텍스트 정제
        df['cleaned_content'] = df['content'].apply(self.clean_text)
        
        # 토큰화
        df['tokens'] = df['cleaned_content'].apply(self.tokenize_text)
        
        return df

# ESG 분석 클래스
class ESGAnalyzer:
    def __init__(self, esg_keywords, esg_weights, bert_model_path=None):
        self.esg_keywords = esg_keywords
        self.esg_weights = esg_weights
        
        # BERT 모델 로드
        if bert_model_path and os.path.exists(bert_model_path):
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
            self.model = BertForSequenceClassification.from_pretrained(bert_model_path)
            self.use_bert = True
        else:
            # 기본 BERT 모델 사용
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
            self.use_bert = False
            logger.warning("사전 훈련된 ESG BERT 모델을 찾을 수 없어 기본 모델을 사용합니다.")
        
        # 감성 분석 파이프라인 설정
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
    def keyword_based_analysis(self, text):
        """키워드 기반 ESG 분석"""
        text = text.lower()
        
        # 각 ESG 카테고리별 키워드 출현 횟수 계산
        esg_scores = {
            'environmental': 0,
            'social': 0,
            'governance': 0
        }
        
        for category, keywords in self.esg_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    esg_scores[category] += 1
        
        return esg_scores
        
    def bert_based_analysis(self, text):
        """BERT 기반 ESG 분석"""
        if not self.use_bert:
            # 사전 훈련된 ESG BERT 모델이 없는 경우 키워드 기반 분석 결과 반환
            return self.keyword_based_analysis(text)
        
        # 텍스트 인코딩
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        # 예측
        outputs = self.model(**inputs)
        predictions = outputs.logits.softmax(dim=1).detach().numpy()[0]
        
        # 예측 결과를 ESG 카테고리에 매핑 (0: 환경, 1: 사회, 2: 지배구조)
        esg_scores = {
            'environmental': float(predictions[0]),
            'social': float(predictions[1]),
            'governance': float(predictions[2])
        }
        
        return esg_scores
        
    def sentiment_analysis(self, text):
        """텍스트 감성 분석"""
        try:
            # TextBlob을 이용한 감성 분석
            sentiment = TextBlob(text).sentiment.polarity
            
            # Hugging Face 트랜스포머 모델을 이용한 감성 분석 (더 정확함)
            if len(text) > 10:  # 너무 짧은 텍스트는 건너뜀
                result = self.sentiment_analyzer(text[:512])[0]  # 최대 512 토큰으로 제한
                
                # 결과를 -1에서 1 사이의 값으로 정규화
                if result['label'] == 'POSITIVE':
                    sentiment = result['score']
                else:
                    sentiment = -result['score']
            
            return sentiment
        except Exception as e:
            logger.error(f"감성 분석 중 오류 발생: {str(e)}")
            return 0
        
    def analyze_data(self, df):
        """데이터프레임 ESG 분석"""
        if df.empty:
            logger.warning("분석할 데이터가 없습니다.")
            return df
        
        # ESG 분석 결과 초기화
        df['environmental_score'] = 0.0
        df['social_score'] = 0.0
        df['governance_score'] = 0.0
        df['esg_total_score'] = 0.0
        df['sentiment_score'] = 0.0
        
        # 각 텍스트에 대한 ESG 분석 수행
        for idx, row in df.iterrows():
            text = row['cleaned_content']
            
            # ESG 분석 (가능하면 BERT 기반, 아니면 키워드 기반)
            if self.use_bert:
                esg_scores = self.bert_based_analysis(text)
            else:
                esg_scores = self.keyword_based_analysis(text)
            
            # 감성 분석
            sentiment = self.sentiment_analysis(text)
            
            # 결과 저장
            df.at[idx, 'environmental_score'] = esg_scores['environmental']
            df.at[idx, 'social_score'] = esg_scores['social']
            df.at[idx, 'governance_score'] = esg_scores['governance']
            df.at[idx, 'sentiment_score'] = sentiment
            
            # 가중치 적용한 총점 계산
            weighted_score = (
                esg_scores['environmental'] * self.esg_weights['environmental'] +
                esg_scores['social'] * self.esg_weights['social'] +
                esg_scores['governance'] * self.esg_weights['governance']
            )
            
            # 감성 점수를 반영한 총점 계산 (긍정적일수록 높은 점수)
            sentiment_factor = 1 + (sentiment * 0.2)  # 감성 효과 20%로 제한
            df.at[idx, 'esg_total_score'] = weighted_score * sentiment_factor
        
        return df

# 결과 시각화 및 보고서 생성 클래스
class ESGReporter:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_summary(self, df):
        """ESG 분석 결과 요약"""
        if df.empty:
            logger.warning("요약할 데이터가 없습니다.")
            return {}
            
        # 소스별 데이터 수
        source_counts = df['source'].value_counts().to_dict()
        
        # ESG 카테고리별 평균 점수
        environmental_avg = df['environmental_score'].mean()
        social_avg = df['social_score'].mean()
        governance_avg = df['governance_score'].mean()
        esg_total_avg = df['esg_total_score'].mean()
        
        # 긍정/부정 비율
        positive_count = len(df[df['sentiment_score'] > 0])
        negative_count = len(df[df['sentiment_score'] < 0])
        neutral_count = len(df[df['sentiment_score'] == 0])
        
        # 상위 ESG 점수 문서
        top_docs = df.nlargest(5, 'esg_total_score')[['source', 'title', 'url', 'esg_total_score']]
        
        # 요약 정보 생성
        summary = {
            'data_count': len(df),
            'source_distribution': source_counts,
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'esg_scores': {
                'environmental': float(environmental_avg),
                'social': float(social_avg),
                'governance': float(governance_avg),
                'total': float(esg_total_avg)
            },
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            },
            'top_documents': top_docs.to_dict('records')
        }
        
        return summary
        
    def create_visualizations(self, df, company_name):
        """ESG 분석 결과 시각화"""
        if df.empty:
            logger.warning("시각화할 데이터가 없습니다.")
            return
            
        # 시각화를 위한 스타일 설정
        plt.style.use('seaborn')
        
        # 1. ESG 카테고리별 점수 분포 (파이 차트)
        esg_scores = {
            'Environmental': df['environmental_score'].mean(),
            'Social': df['social_score'].mean(),
            'Governance': df['governance_score'].mean()
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        wedges, texts, autotexts = ax.pie(
            esg_scores.values(), 
            labels=esg_scores.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=['#66c2a5', '#fc8d62', '#8da0cb']
        )
        ax.set_title(f'ESG 카테고리별 분포 - {company_name}', fontsize=15)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{company_name}_esg_distribution.png", dpi=300)
        plt.close()
        
        # 2. 소스별 ESG 점수 (막대 그래프)
        source_group = df.groupby('source').agg({
            'environmental_score': 'mean',
            'social_score': 'mean',
            'governance_score': 'mean'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        source_group.plot(
            x='source',
            y=['environmental_score', 'social_score', 'governance_score'],
            kind='bar',
            ax=ax,
            color=['#66c2a5', '#fc8d62', '#8da0cb']
        )
        ax.set_title(f'소스별 ESG 점수 - {company_name}', fontsize=15)
        ax.set_xlabel('데이터 소스')
        ax.set_ylabel('평균 점수')
        ax.legend(['환경(E)', '사회(S)', '지배구조(G)'])
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{company_name}_source_esg_scores.png", dpi=300)
        plt.close()
        
        # 3. 감성 분석 결과 (히스토그램)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['sentiment_score'], bins=20, kde=True, ax=ax)
        ax.set_title(f'감성 분석 분포 - {company_name}', fontsize=15)
        ax.set_xlabel('감성 점수 (부정 → 긍정)')
        ax.set_ylabel('빈도')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{company_name}_sentiment_distribution.png", dpi=300)
        plt.close()
        
        # 4. 시간에 따른 ESG 점수 변화 (라인 차트)
        # 날짜 형식 변환
        df['date'] = pd.to_datetime(df['date'])
        
        # 날짜별 그룹화
        time_group = df.groupby(df['date'].dt.date).agg({
            'environmental_score': 'mean',
            'social_score': 'mean',
            'governance_score': 'mean',
            'esg_total_score': 'mean'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        time_group.plot(
            x='date',
            y=['environmental_score', 'social_score', 'governance_score', 'esg_total_score'],
            kind='line',
            marker='o',
            ax=ax
        )
        ax.set_title(f'시간에 따른 ESG 점수 변화 - {company_name}', fontsize=15)
        ax.set_xlabel('날짜')
        ax.set_ylabel('평균 점수')
        ax.legend(['환경(E)', '사회(S)', '지배구조(G)', 'ESG 총점'])
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{company_name}_time_series.png", dpi=300)
        plt.close()
        
        logger.info(f"{company_name} 시각화 파일 생성 완료")
        
    def generate_report(self, df, company_name):
        """ESG 모니터링 보고서 생성"""
        if df.empty:
            logger.warning("보고서를 생성할 데이터가 없습니다.")
            return
            
        # 요약 정보 생성
        summary = self.generate_summary(df)
        
        # 시각화 생성
        self.create_visualizations(df, company_name)
        
        # HTML 보고서 생성
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 보고서 HTML 템플릿
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ESG 모니터링 보고서 - {company_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary-box {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 20px; }}
                .score-container {{ display: flex; justify-content: space-between; }}
                .score-box {{ flex: 1; text-align: center; padding: 15px; margin: 10px; border-radius: 5px; color: white; }}
                .env-box {{ background-color: #66c2a5; }}
                .social-box {{ background-color: #fc8d62; }}
                .gov-box {{ background-color: #8da0cb; }}
                .total-box {{ background-color: #4d4d4d; }}
                .visualization {{ margin: 30px 0; text-align: center; }}
                .viz-img {{ max-width: 100%; height: auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ESG 모니터링 보고서</h1>
                <p><strong>기업명:</strong> {company_name}</p>
                <p><strong>분석 시간:</strong> {report_time}</p>
                <p><strong>데이터 기간:</strong> {summary['date_range']['start']} - {summary['date_range']['end']}</p>
                <p><strong>분석 데이터 수:</strong> {summary['data_count']} 건</p>
                
                <h2>ESG 점수 요약</h2>
                <div class="score-container">
                    <div class="score-box env-box">
                        <h3>환경(E)</h3>
                        <p style="font-size: 24px; font-weight: bold;">{summary['esg_scores']['environmental']:.2f}</p>
                    </div>
                    <div class="score-box social-box">
                        <h3>사회(S)</h3>
                        <p style="font-size: 24px; font-weight: bold;">{summary['esg_scores']['social']:.2f}</p>
                    </div>
                    <div class="score-box gov-box">
                        <h3>지배구조(G)</h3>
                        <p style="font-size: 24px; font-weight: bold;">{summary['esg_scores']['governance']:.2f}</p>
                    </div>
                    <div class="score-box total-box">
                        <h3>ESG 총점</h3>
                        <p style="font-size: 24px; font-weight: bold;">{summary['esg_scores']['total']:.2f}</p>
                    </div>
                </div>
                
                <h2>데이터 소스 분포</h2>
                <div class="summary-box">
                    <table>
                        <tr>
                            <th>소스</th>
                            <th>건수</th>
                        </tr>
        """
        