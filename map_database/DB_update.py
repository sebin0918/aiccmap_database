# 표준 라이브러리
# 표준 라이브러리
from datetime import datetime, timedelta
import urllib.request
import urllib.parse
import json
import re

# 외부 라이브러리
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertModel, BertTokenizer, GPT2LMHeadModel, PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from sentence_transformers import SentenceTransformer
from pykrx import stock as stk
from fredapi import Fred
import yfinance as yf
import torch.nn as nn
import networkx as nx
import torch.nn as nn
import networkx as nx
import pandas as pd
import numpy as np
import numpy as np
import requests
import pymysql
import pickle
import joblib
import torch
import faiss

# NLTK 관련 다운로드
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# NLTK 모듈
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords

# 안전한 텐서 사용
from safetensors.torch import safe_open

# 로깅 및 경고
from transformers import logging as hf_logging
import warnings
import logging

warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()


# 데이터베이스 연결 정보
database_info = open('/scripts/database_info/database_id.txt', 'r').readlines()

sql_file_path = database_info[0].replace('\n', '')
database_host_ip = database_info[1].replace('\n', '')
database_name = database_info[2].replace('\n', '')
database_id = database_info[3].replace('\n', '')
database_passwd = database_info[4].replace('\n', '')
database_charset = database_info[5].replace('\n', '')

# API KEY
API_key = open('/scripts/database_info/api_key.txt', 'r').readlines()

ko_bank_key = API_key[0].replace('\n', '')
fred_api_key = API_key[1].replace('\n', '')

# 네이버 API 클라이언트 ID와 시크릿
client_id = API_key[2].replace('\n', '')
client_secret = API_key[3].replace('\n', '')

# ================================================================== Function ==================================================================


# per pbr 계산 함수
def get_per_pbr_df(ticker_symbol, start_date, end_date):
    # 주식 데이터 다운로드
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(start=start_date, end=end_date)

    # 재무 데이터 다운로드
    financials = ticker.financials
    balance_sheet = ticker.balance_sheet

    # EPS 및 BVPS 계산
    try:
        net_income = financials.loc['Net Income'].iloc[0]  # 최신 데이터 사용
        shares_outstanding = ticker.info.get('sharesOutstanding', None)
        if shares_outstanding is None:
            raise ValueError("Shares outstanding not available")
        eps = net_income / shares_outstanding

        total_assets = balance_sheet.loc['Total Assets'].iloc[0]
        total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]  # 대체 가능한 키 사용
        book_value = total_assets - total_liabilities
        bvps = book_value / shares_outstanding
    except Exception as e:
        print(f"Error calculating EPS or BVPS: {e}")
        return pd.DataFrame()  # 빈 데이터프레임 반환

    # 기간 동안의 주가 데이터를 기반으로 PER 및 PBR 계산
    per_list = []
    pbr_list = []
    dates = []

    for date, row in data.iterrows():
        avg_price = row['Close']
        per = avg_price / eps
        pbr = avg_price / bvps
        per_list.append(per)
        pbr_list.append(pbr)
        dates.append(date)

    # 데이터프레임 생성
    result_df = pd.DataFrame({
        'Date': dates,
        'PER': per_list,
        'PBR': pbr_list
    })

    result_df.set_index('Date', inplace=True)
    return result_df

# 날짜 형식 변환 함수 
# 2024Q1  -> 20240101 
# 2024    -> 20240101
# 202401  -> 20240101
def check_time_data(check_time) :
    if 'Q1' in check_time :
        check_time = f'{check_time[:4]}0101'
    elif 'Q2' in check_time :
        check_time = f'{check_time[:4]}0401'
    elif 'Q3' in check_time :
        check_time = f'{check_time[:4]}0701'
    elif 'Q4' in check_time :
        check_time = f'{check_time[:4]}1001'
    elif len(check_time) <= 4 :
        check_time = f'{check_time}0101'
    elif len(check_time) <= 6 :
        check_time = f'{check_time}01'
    return check_time

def qchange(year, month) :
    if str(month) in ['01', '02', '03'] :
        month = 'Q1'
    elif str(month) in ['04', '05', '06'] :
        month = 'Q2'
    elif str(month) in ['07', '08', '09'] :
        month = 'Q3'
    elif str(month) in ['10', '11', '12'] :
        month ='Q4'
    return f'{year}{month}'

# 날짜 형식 변환 함수 (20240905 -> 2024-09-05)
def convert_date_format(date_str):
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

# URL 요청 함수: API에 요청을 보내고 응답을 반환
def getRequestUrl(url):
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", client_id)
    req.add_header("X-Naver-Client-Secret", client_secret)
    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            return response.read().decode('utf-8')
        else:
            return None
    except Exception as e:
        print(e)
        return None

# 네이버 뉴스 검색 API를 사용하여 뉴스를 검색하는 함수
def searchNaverNews(query, from_date, to_date, display=10, start=1):
    base_url = "https://openapi.naver.com/v1/search/news.json"
    query = urllib.parse.quote(query)
    url = f"{base_url}?query={query}&display={display}&start={start}&sort=date&startDate={from_date}&endDate={to_date}"
    
    response = getRequestUrl(url)
    if response is None:
        return None
    
    return json.loads(response)

# 네이버 뉴스 본문을 크롤링하는 함수
def fetchNaverNewsContent(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 네이버 뉴스 본문 추출 (주어진 HTML 구조에 맞게 수정)
            article_body = soup.find('article', {'id': 'dic_area', 'class': 'go_trans _article_content'})
            
            if article_body:
                # 불필요한 태그 제거 (예: 스크립트, 스타일 등)
                for unwanted in article_body(['script', 'style']):
                    unwanted.extract()
                return article_body.get_text(strip=True)
            else:
                return ""
        else:
            return ""
    except Exception as e:
        return str(e)

def getFilteredNews(query, date, keywords=None, display=500):
    # keywords가 주어지지 않으면 빈 리스트로 설정
    if keywords is None:
        keywords = ['HTTP', 'http', '404 ERROR', '500 ERROR', '페이지를 찾지못했습니다.', 'HTTPSConnectionPool', "!eo$"]
    
    news_items = searchNaverNews(query, 20200917, date, display)
    
    if news_items:
        parsed_news = []
        for item in news_items['items']:
            try:
                title = BeautifulSoup(item['title'], 'html.parser').get_text()
                link = item['link']
                description = BeautifulSoup(item['description'], 'html.parser').get_text()
                
                # 원문 기사 크롤링
                content = fetchNaverNewsContent(link)
                summary_content = textrank_summary(content)
                # title, link, description, content 중 하나라도 비어 있으면 해당 기사를 넘기기
                if not title or not link or not description or not content:
                    continue
                
                # 키워드 리스트가 비어 있지 않으면 필터링
                if keywords and any(keyword in content for keyword in keywords):
                    continue
                
                parsed_news.append({
                    'title': title,
                    'link': link,
                    'description': description,
                    'content': content,
                    'summary_content': summary_content
                })
            
            except Exception as e:
                continue
        
        return parsed_news
    else:
        return []

def sentence_similarity(sent1, sent2, stopwords=None):
	# 불용어가 주어지지 않으면 빈 리스트로 초기화
	if stopwords is None:
		stopwords = []
		
	# 문장을 모두 소문자로 변환
	sent1 = [word.lower() for word in sent1 if word not in stopwords]
	sent2 = [word.lower() for word in sent2 if word not in stopwords]
	
	# 두 문장에서 모든 단어를 모두 중복을 제거한 집합 생성
	all_words = list(set(sent1 + sent2))
	
	# 각 문장의 단어 등장 횟수를 기록할 벡터 초기화
	vector1 = [0] * len(all_words)
	vector2 = [0] * len(all_words)
	
	# 첫 번째 문장의 단어 등장 횟수 기록
	for word in sent1:
		if word in stopwords:
			continue # 불용어 무시
		vector1[all_words.index(word)] += 1
	
	# 두 번째 문장의 단어 등장 횟수 기록
	for word in sent2:
		if word in stopwords:
			continue
		vector2[all_words.index(word)] += 1
		
		# 코사인 유사도 계산을 위해 벡터를 이용하여 측정
		return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):

	# 문장 간 유사도 행렬 초기화
	similarity_matrix = np.zeros((len(sentences), len(sentences)))
	
	# 모든 문장 쌍에 대해 유사도 계산
	for idx1 in range(len(sentences)):
		for idx2 in range(len(sentences)):
			# 같은 문장인 경우 계산하지 않음
			if idx1 == idx2:
				continue
				
			# 문장 간 유사도 게산하여 행렬에 할당
			similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
			
	return similarity_matrix

def textrank_summary(content,  num_sentences=3):
    # 입력 텍스트를 문장 단위로 구분
    sentences = sent_tokenize(content)

    stop_words = ['을', '를', '이', '가', '은', '는', '에', '의', '과', '와', '한', '들', '의']
    sentences = [word for word in sentences if word not in stop_words]
    
    # 문장 간 유사도 행렬 생성
    similarity_matrix = build_similarity_matrix(sentences, stop_words)
    
    # 페이지랭크 알고리즘을 사용하여 각 문장의 점수 계산
    scores = nx.pagerank(nx.from_numpy_array(similarity_matrix))
    
    # 문장을 페이지랭크 점수에 따라 내림차순으로 정렬
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # 상위 num_sentences 만큼의 문장을 선택하여 요약 생성
    summary = ' '.join([sentence for score, sentence in ranked_sentences[:num_sentences]])
    comparison_texts = [content]
    for comp_text in comparison_texts:
        sim = sentence_similarity(summary, comp_text)
        if sim >= 0.95:
            return None
    return summary

# 모델과 토크나이저 설정
class KCBERTModel(nn.Module):
    def __init__(self, model_name='beomi/kcbert-base'):
        super(KCBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 이진 분류

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # BERT의 [CLS] 토큰 출력
        return self.classifier(pooled_output)

# 모델 로드 함수
def load_kcbert_model(model_path):
    model = KCBERTModel()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"모델 '{model_path}'이 CPU로 로드되었습니다.")
    except Exception as e:
        print(f"모델을 로드하는 중 오류가 발생했습니다: {e}")
    return model

# 텍스트 분류 함수
def predict_classification(model, tokenizer, text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predicted_label = torch.argmax(outputs, dim=1).item()

    label_map = {0: '악재', 1: '호재'}
    return label_map[predicted_label]

# 예측을 데이터프레임에 추가하는 함수
def apply_model_predictions(df, model, tokenizer, text_column='content'):
    df['Kc_BERT'] = df[text_column].apply(lambda text: predict_classification(model, tokenizer, text))
    return df

# 모든 종목의 데이터프레임 처리
def process_stock_dataframes(df_dict, models, tokenizer):
    processed_dfs = [apply_model_predictions(df_dict[stock], models[stock], tokenizer) for stock in df_dict]
    return pd.concat(processed_dfs, ignore_index=True)

# 뉴스 분류 함수 (KF-DEBERTa 모델)
def newsClassification_stock_predict(news, stock):
    model_path = "kakaobank/kf-deberta-base"
    safetensors_coin_file = "./news_model/news_{stock}_KF-DeBERTa.safetensors"
    
    try:
        coin_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        with safe_open(safetensors_coin_file, framework="pt", device="cpu") as f:
            state_dict_coin = {key: f.get_tensor(key) for key in f.keys()}
        coin_model.load_state_dict(state_dict_coin, strict=False)
    except Exception as e:
        return None

    try:
        inputs = tokenizer(news, return_tensors='pt')
        outputs = coin_model(**inputs)
        logit = outputs.logits
        predict = logit.argmax(dim=-1).item()
        return "호재" if predict == 1 else "악재"
    except Exception as e:
        return None
    
# 긍정/부정 단어 검증 및 분류 함수
def classify_news(content):
    po = sum(1 for word in positive_words if word in content)
    ne = sum(-1 for word in negative_words if word in content)
    
    if po > 0 or ne < 0:
        verification_value = po + ne
    else:
        verification_value = 5000

    if verification_value == 5000 or verification_value == 0:
        return '부정확'
    elif verification_value > 0:
        return '호재'
    else:
        return '악재'
    
# 3개의 분류 방법 결과 확인
def final_classification(row):
    if row['classification'] == row['Kc_BERT'] or row['classification'] == row['KF-DEBERTa']:
        return row['classification']
    else:
        return '부정확'
        
# txt 파일 로드
def load_words(file_path):
    encodings = ['utf-8', 'cp949', 'euc-kr']  # 시도할 인코딩 리스트
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return [line.strip() for line in f if line.strip()]
        except UnicodeDecodeError:
            print(f"'{encoding}' 인코딩으로 파일을 읽을 수 없습니다. 다른 인코딩을 시도합니다.")
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {file_path}")
            break
    print("모든 인코딩 시도가 실패했습니다.")
    return []

def update_title_with_classification(row):
    if row['final_classification'] == '호재':
        if row['query'] == '삼성전자':
            return f"{row['title']}!eo$삼성전자 호재입니다."
        elif row['query'] == '비트코인':
            return f"{row['title']}!eo$비트코인 호재입니다."
        elif row['query'] == '애플':
            return f"{row['title']}!eo$애플 호재입니다."
    elif row['final_classification'] == '악재':
        if row['query'] == '삼성전자':
            return f"{row['title']}!eo$삼성전자 악재입니다."
        elif row['query'] == '비트코인':
            return f"{row['title']}!eo$비트코인 악재입니다."
        elif row['query'] == '애플':
            return f"{row['title']}!eo$애플 악재입니다."
    return row['title']  # 호재/악재가 아니면 기존 제목 유지

# ================================================================== stock market news ===================================================================
def getRequestUrl(url):
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", client_id)
    req.add_header("X-Naver-Client-Secret", client_secret)
    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            return response.read().decode('utf-8')
        else:
            print(f"HTTP Error {response.getcode()}: {response.reason}")
            return None
    except Exception as e:
        print(e)
        return None


def sentence_similarity(sent1, sent2, stopwords=None):
    # 불용어가 주어지지 않으면 빈 리스트로 초기화
    if stopwords is None:
        stopwords = []

    # 문장을 모두 소문자로 변환
    sent1 = [word.lower() for word in sent1 if word not in stopwords]
    sent2 = [word.lower() for word in sent2 if word not in stopwords]

    # 두 문장에서 모든 단어를 모두 중복을 제거한 집합 생성
    all_words = list(set(sent1 + sent2))
    
    # 각 문장의 단어 등장 횟수를 기록할 벡터 초기화
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # 첫 번째 문장의 단어 등장 횟수 기록
    for word in sent1:
        if word in stopwords:
            continue # 불용어 무시
        vector1[all_words.index(word)] += 1
    
    # 두 번째 문장의 단어 등장 횟수 기록
    for word in sent2:
        if word in stopwords:
            continue
        vector2[all_words.index(word)] += 1
        
        # 코사인 유사도 계산을 위해 벡터를 이용하여 측정
        return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):

    # 문장 간 유사도 행렬 초기화
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    # 모든 문장 쌍에 대해 유사도 계산
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            # 같은 문장인 경우 계산하지 않음
            if idx1 == idx2:
                continue
            
            # 문장 간 유사도 게산하여 행렬에 할당
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def searchNaverNews(query, display=10, start=1):
    base_url = 'https://openapi.naver.com/v1/search/news.json'
    query    = urllib.parse.quote(query)
    url      = f"{base_url}?query={query}&display={display}&start={start}&sort=date"

    response = getRequestUrl(url)
    if response is None:
        return None

    return json.loads(response)


def textrank_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)

    stop_words = ['을', '를', '이', '가', '은', '는', '에', '의', '과', '와', '한', '들', '의']
    sentences  = [word for word in sentences if word not in stop_words]

    if len(sentences) < 2:
        return ' '.join(sentences)

    similarity_matrix = build_similarity_matrix(sentences, stop_words)
    scores            = nx.pagerank(nx.from_numpy_array(similarity_matrix))
    ranked_sentences  = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary           = ' '.join([sentence for score, sentence in ranked_sentences[:num_sentences]])

    return summary

def economic_news_search(num_days=30, display=100):
    all_filtered_news = []
    collected_data    = set()

    # 확장된 증시 관련 키워드 목록
    keywords = [
        '증시', '코스피', '코스닥', '주식', '상장', '상장폐지', '배당', '배당금', '주가', '시가총액',
        'PER', 'PBR', 'EPS', '매출', '순이익', '상승', '하락', '변동성', '투자', '매수', '매도', '공매도',
        '기관투자자', '개인투자자', 'ETF', '선물', '옵션', '지수', '거래량', '상한가', '하한가', '상승장',
        '하락장', '공시', '기업공개', 'IPO', '시장동향', '경제지표', '금리', '인플레이션', '디플레이션',
        '유동성', '재무제표', '주주총회', '배당수익률', '채권', '펀드', '헤지펀드', '알고리즘', '기술적분석',
        '기본적분석', '리스크', '포트폴리오', '다각화', '시장점유율', '신규상장', '기업실적', '경영전략',
        '재무상태', '부채비율', '유보율', '자본금', '배당정책', '주식시장', '금융시장', '경제성장', '국제금융',
        '환율', '원화', '달러', '유로', '엔화', '금', '은', '원자재', '에너지', '반도체', 'IT', '바이오',
        '헬스케어', '제약', '자동차', '건설', '부동산', '소비재', '필수소비재', '이차전지', '전기차', '친환경',
        '재생에너지', '스마트폰', '디지털', '블록체인', '암호화폐', '비트코인', '이더리움', '금융정책', '경제정책',
        '금융위기', '코로나19', '백신', '금리인상', '금리인하', '채권시장', '부동산시장', '인플레이션율',
        '디플레이션율', '경제성장률', 'GDP', '실업률', '소비자물가지수', '생산자물가지수', '수출', '수입',
        '경상수지', '자본수지', '무역수지', '금융수지', '경기순환', '거시경제', '미시경제', '금융기관', '은행',
        '증권사', '자산관리', '재테크', '부동산투자', '주식투자', '펀드투자', '채권투자', 'ETF투자', '리츠',
        '벤처캐피탈', '스타트업', '핀테크', '모바일뱅킹', '온라인증권', '로보어드바이저', '디지털자산', '가상자산',
        '스마트컨트랙트', '탈중앙화', '디파이', 'NFT', '메타버스', 'AI투자', '빅데이터', '클라우드컴퓨팅',
        'IoT', '5G', '자율주행', '친환경차', '스마트시티', '헬스케어기술', '바이오테크', '제로에너지건축',
        '그린에너지', '재생가능에너지', '탄소중립', 'ESG투자', '사회책임투자', '지속가능투자', '투자전략'
    ]

    exclude_keywords = ['기자', '?', "앵커", '투자', '운용사', '괜찮아요', 'http', '신진대사', '체질', '날씨', '기온']
    today = datetime.today()
    formatted_date = f"{today.month}월 {today.day}일"
    # 쿼리 리스트 정의 (내부에서 고정된 쿼리 목록 사용, 증시 대표 키워드로 추가)
    queries = [
    '증시', '미국증시', '한국증시', '나스닥', '다우지수', 'S&P500', '니케이', '상해종합', 'FTSE100', 'DAX30', '항셍지수',
    '유가', '천연가스', '미국채', '유럽경제', '기술주', '헬스케어', '핀테크', 'ESG', '인공지능', '비트코인',
    '블록체인', '암호화폐', '달러', '원유', '금', '은', '구리', '철광석', '리튬', '배터리', '전기차', '테슬라',
    '애플', '마이크로소프트', '아마존', '구글', '페이스북', '메타버스', '코로나19', '백신', '반도체', '5G',
    'AI', '로봇공학', '양자 컴퓨팅', '사이버 보안', '자율주행', '클라우드컴퓨팅', '대선', '재생에너지',
    '친환경차', '삼성전자', "HBM", "BDSPN", "파운드리"
    ]
    queries_with_date = [f"{formatted_date} {query}" for query in queries]

    for query in queries_with_date:
        filtered_news = searchNaverNews(query, display=display)
        if filtered_news and 'items' in filtered_news:
            for item in filtered_news['items']:
                title = BeautifulSoup(item['title'], 'html.parser').get_text()
                description = BeautifulSoup(item['description'], 'html.parser').get_text()
                content = description

            # 제외할 키워드가 포함된 뉴스는 건너뛰기
            if any(exclude in title for exclude in exclude_keywords) or any(exclude in content for exclude in exclude_keywords):
                continue

            # 제목, 설명, 본문 중 하나라도 중복된 경우 건너뛰기
            if title in collected_data or description in collected_data or content in collected_data:
                continue

            if not '증시' in content:
                continue

            # 키워드와 겹치는 단어 수 계산
            matched_keywords = [keyword for keyword in keywords if keyword in title or keyword in content]
            if len(matched_keywords) < 3:  # 키워드가 N개 이상 겹치는 경우에만 가져옴
                continue

            link = item['link']
            news_data = {
                'title': title,
                'description': description,
                'link': link,
                'Date': datetime.strptime(item['pubDate'], '%a, %d %b %Y %H:%M:%S +0900').strftime('%Y-%m-%d'),
                'content': description,
            }

            news_data['summary'] = textrank_summary(news_data['description'])
            all_filtered_news.append(news_data)
            collected_data.update([title, description, content])

    return None if not all_filtered_news else pd.DataFrame(all_filtered_news)

def get_response(user_query_pre, top_k=3, max_new_tokens=150, try_count=False):
    today = datetime.today()
    formatted_date = f"{today.month}월 {today.day}일"

    user_query =  formatted_date + user_query_pre

    # 1. 뉴스 데이터 불러오기 및 전처리
    df = economic_news_search()  # 뉴스 데이터를 가져오는 함수
    df = df.dropna(subset=['summary'])  # 요약이 없는 데이터는 제거
    texts = df['summary'].tolist()  # 요약 부분만 리스트로 변환
    def remove_urls(text):
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)

    texts = [remove_urls(text) for text in texts]
    # 2. 한국어 GPT 모델 설정
    generator_model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gpt2')

    # padding 토큰 설정 (GPT 모델에서는 eos_token을 padding으로 사용)
    tokenizer.pad_token = tokenizer.eos_token
    generator_model.config.pad_token_id = tokenizer.eos_token_id

    # pad_token이 없는 경우, 새로운 pad_token 추가
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        generator_model.resize_token_embeddings(len(tokenizer))

    # 3. SentenceTransformer 모델을 사용한 임베딩 모델 설정
    embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    # 4. FAISS 인덱스 생성 및 추가
    document_embeddings = embedding_model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    document_embeddings = np.array(document_embeddings).astype('float32')
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(document_embeddings)

    # 5. 질의에 맞는 관련 뉴스를 검색하는 함수
    def get_relevant_summaries(query, top_k=3):
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        distances, indices = index.search(query_embedding, top_k)
        relevant_summaries = [texts[idx] for idx in indices[0]]
        return relevant_summaries

    # 6. 무조건 CPU 장치 설정
    device = torch.device('cpu')
    generator_model.to(device)

    # 7. 답변 생성
    relevant_summaries = get_relevant_summaries(user_query, top_k)
    context = " ".join(relevant_summaries)
    prompt = f"뉴스 요약: {context}\n질문: {user_query}\n답변: "

    inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
    attention_mask = inputs['attention_mask']
    outputs = generator_model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split('답변:')[-1].strip()
    def process_response(answer):
        # "괜찮아요." 제거 및 세 번째 "다."에서 자르기
        answer = answer.replace("괜찮아요.\n", "")
        split_response = answer.split("다.")[:3]
        final_response = "다.".join(split_response) + "다."

        # 첫 번째 한글이 등장하는 위치를 찾고 그 이후의 텍스트 반환
        korean_text_match = re.search(r'[가-힣]', final_response)
        if korean_text_match:
            korean_text_start = korean_text_match.start()
            return final_response[korean_text_start:]  # 한글 이후 부분 반환
        else:
            return final_response  # 한글이 없으면 전체 텍스트 반환
    final_response = process_response(answer)


    print("상세 응답:", final_response)

    # 반환값이 http로 시작하는 경우 다시 실행
    if (final_response in ("", None) or user_query_pre not in final_response) and not try_count:
        time.sleep(100)
        return get_response(user_query, top_k, max_new_tokens, try_count=True)
    if (final_response in ("", None) or user_query_pre not in final_response) and try_count:
        return process_response(context)

    return final_response

user_query_pre = " 증시"
response = get_response(user_query_pre)
f_stock_market_text = open('./stock_market/stock_market.txt', 'w', encoding='utf-8')
f_stock_market_news = f_stock_market_text.write(response)
f_stock_market_text.close()

# ================================================================== News data ==================================================================

# 쿼리 리스트와 현재 날짜 설정
queries = {
    'samsung': '삼성전자',
    'apple': '애플',
    'bitcoin': '비트코인'
}

date = datetime.now().strftime("%Y%m%d")  # 현재 날짜를 "YYYYMMDD" 형식으로 변환

# 각 쿼리로 뉴스 데이터를 가져오고 'query' 컬럼 추가
samsung_news_data = getFilteredNews(queries['samsung'], date, display=100)
apple_news_data = getFilteredNews(queries['apple'], date, display=100)
bitcoin_news_data = getFilteredNews(queries['bitcoin'], date, display=100)

# 각 데이터프레임 생성 및 'query' 컬럼 추가
samsung_df = pd.DataFrame(samsung_news_data)
samsung_df['query'] = queries['samsung']

apple_df = pd.DataFrame(apple_news_data)
apple_df['query'] = queries['apple']

bitcoin_df = pd.DataFrame(bitcoin_news_data)
bitcoin_df['query'] = queries['bitcoin']

# 모델 경로와 토크나이저 설정
model_paths = {
    'samsung': './news_model/news_samsung_kcbert_model.pth',
    'apple': './news_model/news_apple_kcbert_model.pkl',
    'bitcoin': './news_model/news_bitcoin_kcbert_model.pkl'
}

# 모델과 토크나이저 로드
models = {stock: load_kcbert_model(path) for stock, path in model_paths.items()}
tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')

# 데이터프레임 처리
df_dict = {'samsung': samsung_df, 'apple': apple_df, 'bitcoin': bitcoin_df}
final_df = process_stock_dataframes(df_dict, models, tokenizer)

# 각 주식 이름에 대해 뉴스 분류 결과를 추가
stocks = ['apple', 'bitcoin', 'samsung']
dfs = {'apple': apple_df, 'bitcoin': bitcoin_df, 'samsung': samsung_df}

for stock in stocks:
    df = dfs[stock]
    df['KF-DEBERTa'] = df['content'].apply(lambda x: newsClassification_stock_predict(x, stock))

# 데이터프레임 병합
merged_df = pd.concat([apple_df, bitcoin_df, samsung_df], ignore_index=True)

# 중복된 뉴스 제거 ('title', 'link', 'description', 'content', 'summary_content' 중 하나라도 같으면 제거)
merged_df.drop_duplicates(subset=['title', 'link', 'description', 'content', 'summary_content'], inplace=True)

positive_words = load_words('./news_model/positive_words_self.txt')
negative_words = load_words('./news_model/negative_words_self.txt')

# 기존 classification 컬럼 추가
merged_df['classification'] = merged_df['content'].apply(classify_news)

# 최종 결과를 담는 final_classification 컬럼 추가
merged_df['final_classification'] = merged_df.apply(final_classification, axis=1)

merged_df['title'] = merged_df.apply(update_title_with_classification, axis=1)

df_news = merged_df[['title', 'summary_content', 'link', 'final_classification']]
df_news.rename(columns={'title' : 'news_title', 'summary_content' : 'news_simple_text', 'link' : 'news_link', 'final_classification' : 'news_classification'}, inplace=True)
df_news = df_news[df_news['news_classification'] != '부정확']
df_news.loc[df_news['news_classification'] == '호재', 'news_classification'] = 0
df_news.loc[df_news['news_classification'] == '악재', 'news_classification'] = 1


print('============= stock data API Start =============')
print('Start time : ', datetime.today())

# 날짜 형식

month_date = str(datetime.now().month)
day_date = str(datetime.now().day)
if len(month_date) == 1 :
    month_date = f'0{month_date}'
if len(day_date) == 1 :
    day_date = f'0{day_date}'

up_date = datetime.now() - timedelta(61)
up_month, up_day = up_date.month, up_date.day
if len(str(up_month)) == 1 :
    up_month = f'0{up_month}'
if len(str(up_day)) == 1 :
    up_day = f'0{up_day}'

start_day = f'{up_date.year}-{up_month}-{up_day}'
end_day = f'{datetime.now().year}-{month_date}-{day_date}' #'2024-08-28'

start_date = datetime(int(up_date.year), int(up_month), int(up_day))
#end_date = datetime(int(datetime.now().year), int(month_date), int(day_date))
end_date = datetime.now()
date_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
df_len = [] # 데이터 길이를 조절하기 위한 리스트


# ************************************** tb_stock **************************************
start_date = datetime.now() - timedelta(61)
end_date = datetime.now()

# 데이터 다운로드 및 전처리
ticker = '005930.KS'
samsung = yf.Ticker(ticker)
samsung_stock = samsung.history(start=start_date, end=end_date)
samsung_stock.rename(columns={'Close': 'sc_ss_stock'}, inplace=True)
samsung_stock.reset_index(inplace=True)

ticker = 'AAPL'
apple = yf.Ticker(ticker)
apple_stock = apple.history(start=start_date, end=end_date)
apple_stock.rename(columns={'Close': 'sc_ap_stock'}, inplace=True)
apple_stock.reset_index(inplace=True)

ticker = 'BTC-USD'
bitcoin = yf.Ticker(ticker)
bitcoin_stock = bitcoin.history(start=start_date, end=end_date)
bitcoin_stock.rename(columns={'Close': 'sc_coin'}, inplace=True)
bitcoin_stock.reset_index(inplace=True)

# 데이터프레임 생성
samsung_stock['fd_date'] = samsung_stock['Date'].dt.date
apple_stock['fd_date'] = apple_stock['Date'].dt.date
bitcoin_stock['fd_date'] = bitcoin_stock['Date'].dt.date

# 데이터프레임 결합
stock_df = pd.DataFrame({
    'fd_date': pd.date_range(start=start_date, end=end_date),
    'sc_ss_stock': samsung_stock.set_index('fd_date')['sc_ss_stock'],
    'sc_ap_stock': apple_stock.set_index('fd_date')['sc_ap_stock'],
    'sc_coin': bitcoin_stock.set_index('fd_date')['sc_coin']
}).fillna(method='ffill')

# ************************************** stock predict **************************************

# XGBoost 예측 함수
def xgboost_predict(stock, df, model_filename):
    stock_dataset = df[['fd_date', stock]].set_index('fd_date')
    start_day = 59
    forecast_days = 14

    raw = [stock_dataset.shift(i) for i in range(start_day, 0, -1)]
    sum_df = pd.concat(raw, axis=1).dropna()

    train = sum_df.values
    X = train[:, :-1]

    model = joblib.load(model_filename)

    data_in = stock_dataset[-start_day:].values.flatten().reshape(1, -1)

    forecast_results = []
    for _ in range(forecast_days):
        prediction = model.predict(data_in)
        forecast_results.append(prediction[0])
        new_data = np.append(data_in[0][1:], prediction)
        data_in = new_data.reshape(1, -1)

    forecast_dates = pd.date_range(start=stock_dataset.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame(forecast_results, index=forecast_dates, columns=['Forecast'])

    return forecast_df

# SARIMA 예측 함수
def sarima_predict_up_to_date(stock, model_path, stock_df, lookback_days=60, steps_ahead=14):
    latest_date = stock_df['fd_date'].max()
    input_date = pd.to_datetime(latest_date)

    df_filtered = stock_df[stock_df['fd_date'] <= latest_date].set_index('fd_date').asfreq('D').fillna(method='ffill')

    with open(model_path, 'rb') as pkl_file:
        loaded_model = pickle.load(pkl_file)

    last_observed = df_filtered[stock].iloc[-lookback_days:]
    future_pred_val = []

    while len(future_pred_val) < steps_ahead:
        pred = loaded_model.get_forecast(steps=steps_ahead)
        pred_mean = pred.predicted_mean.cumsum() + last_observed.iloc[-1]
        future_pred_val.extend(pred_mean)
        last_observed = pd.concat([last_observed, pred_mean]).iloc[-lookback_days:]

    future_pred_dates = pd.date_range(start=input_date + timedelta(days=1), periods=len(future_pred_val), freq='D')
    forecast_df = pd.DataFrame(future_pred_val, index=future_pred_dates, columns=['Forecast'])
    return forecast_df

# 최종 데이터프레임 생성 함수
def create_final_dataframe(xgb_df, sarima_ap_df, sarima_bit_df):
    final_df = pd.DataFrame({
        'sp_date': xgb_df.index,
        'sp_ss_predict': xgb_df['Forecast'],
        'sp_ap_predict': sarima_ap_df['Forecast'],
        'sp_bit_predict': sarima_bit_df['Forecast']
    })
    return final_df

# 예측 수행
xgb_predict_df = xgboost_predict('sc_ss_stock', stock_df, './stock_model/regression_stock_samsung_XGBoost.pkl')
sarima_ap_predict_df = sarima_predict_up_to_date('sc_ap_stock', './stock_model/regression_apple_sarima.pkl', stock_df)
sarima_bit_predict_df = sarima_predict_up_to_date('sc_coin', './stock_model/regression_bitcoin_sarima.pkl', stock_df)

# 최종 데이터프레임 생성
df_stock_predict = create_final_dataframe(xgb_predict_df, sarima_ap_predict_df, sarima_bit_predict_df)
print(df_stock_predict)

# ************************************** tb_main_economic_index **************************************
# NASDAQ
nasdaq = yf.download('^IXIC', start=start_day, end='2024-12-31')
nasdaq.rename(columns={'Close': 'mei_nasdaq'}, inplace=True)
nasdaq = nasdaq['mei_nasdaq']

# S&P 500
snp500 = yf.download('^GSPC', start=start_day, end='2024-12-31')
snp500.rename(columns={'Close': 'mei_sp500'}, inplace=True)
snp500 = snp500['mei_sp500']

# Dow Jones Industrial Average (DJI)
dow = yf.download('^DJI', start=start_day, end='2024-12-31')
dow.rename(columns={'Close': 'mei_dow'}, inplace=True)
dow = dow['mei_dow']

# KOSPI
kospi = yf.download('^KS11', start=start_day, end='2024-12-31')
kospi.rename(columns={'Close': 'mei_kospi'}, inplace=True)
kospi = kospi['mei_kospi']

# Gold, Oil
today_date = datetime.today()-timedelta(1)
days_passed = (today_date - start_date).days

# Gold
gold = yf.Ticker('GC=F')
gold_data = gold.history(period='max').tail(days_passed)
gold_data.rename(columns={'Close' : 'mei_gold'}, inplace=True)
gold_data = gold_data['mei_gold']
gold_data = gold_data.reset_index(drop=False)
gold_data['Date'] = gold_data['Date'].astype(str).map(lambda x : x[:10])
gold_data['Date'] = pd.to_datetime(gold_data['Date'])
gold_data.set_index(keys='Date', inplace=True)

# Oil
oil = yf.Ticker('BZ=F')
oil_data = oil.history(period='max').tail(days_passed)
oil_data.rename(columns={'Close' : 'mei_oil'}, inplace=True)
oil_data = oil_data['mei_oil']
oil_data = oil_data.reset_index(drop=False)
oil_data['Date'] = oil_data['Date'].astype(str).map(lambda x : x[:10])
oil_data['Date'] = pd.to_datetime(oil_data['Date'])
oil_data.set_index(keys='Date', inplace=True)

# Exchange Rate
dollar_to_won = yf.download('KRW=X', start_day)
dollar_to_won.rename(columns={'Close' : 'mei_ex_rate'}, inplace=True)
dollar_to_won = dollar_to_won['mei_ex_rate']

# main economic index table insert
tb_main_economic_index_df_list = [nasdaq, snp500, dow, kospi, gold_data, oil_data, dollar_to_won]

main_economic_index_df = date_df.copy()
for i in range(len(tb_main_economic_index_df_list)) :
    main_economic_index_df = main_economic_index_df.join(tb_main_economic_index_df_list[i])

main_economic_index_df = main_economic_index_df.reset_index().rename(columns={'index' : 'fd_date'})
main_economic_index_df.fillna(method='ffill', inplace=True)
main_economic_index_df['fd_date'] = main_economic_index_df['fd_date'].astype(str).map(lambda x : x[:10])


# ************************************** tb_korea_economic_indicator **************************************
data_name_ko = ['국내 총 생산',
            'M2 통화공급 (말잔)',
            'M2 통화공급 (평잔)',
            '중앙은행 기준금리',
            '생산자물가지수',
            '수입물가지수',
            '소비자물가지수',
            '수입',
            '수출',
            '소비자심리지수',
            '기업경기실사지수']

data_name = ['kei_gdp',
            'kei_m2_end',
            'kei_m2_avg',
            'kei_fed_rate',
            'kei_ppi',
            'kei_ipi',
            'kei_cpi',
            'kei_imp',
            'kei_exp',
            'kei_cs',
            'kei_bsi']

api_link = [f'/200Y102/Q/{qchange(up_date.year-1, up_month)}/{qchange(datetime.now().year, month_date)}/10111',
            f'/101Y007/M/{up_date.year-1}{up_month}/{datetime.now().year}{month_date}/BBIA00',
            f'/101Y008/M/{up_date.year-1}{up_month}/{datetime.now().year}{month_date}/BBJA00',
            f'/722Y001/M/{up_date.year-1}{up_month}/{datetime.now().year}{month_date}/0101000',
            f'/404Y014/M/{up_date.year-1}{up_month}/{datetime.now().year}{month_date}/*AA',
            f'/401Y015/M/{up_date.year-1}{up_month}/{datetime.now().year}{month_date}/*AA/W',
            f'/901Y009/M/{up_date.year-1}{up_month}/{datetime.now().year}{month_date}/0',
            f'/403Y003/M/{up_date.year-1}{up_month}/{datetime.now().year}{month_date}/*AA',
            f'/403Y001/M/{up_date.year-1}{up_month}/{datetime.now().year}{month_date}/*AA',
            f'/511Y002/M/{up_date.year-1}{up_month}/{datetime.now().year}{month_date}/FME/99988',
            f'/512Y007/M/{up_date.year-1}{up_month}/{datetime.now().year}{month_date}/AA/99988']

all_data = []
all_time = []
for i in range(len(api_link)) :
    value_time = []
    value_data = []
    search_url = f'https://ecos.bok.or.kr/api/StatisticSearch/{ko_bank_key}/xml/kr/1/2{api_link[i]}'

    search_respons = requests.get(search_url)
    search_xml = search_respons.text
    search_soup = BeautifulSoup(search_xml, 'xml')
    total_val = search_soup.find('list_total_count')
    url = f'https://ecos.bok.or.kr/api/StatisticSearch/{ko_bank_key}/xml/kr/1/{total_val.text}{api_link[i]}'
    
    respons = requests.get(url)
    title_xml = respons.text
    title_soup = BeautifulSoup(title_xml, 'xml') 
    value_d = title_soup.find_all('DATA_VALUE')
    value_t = title_soup.find_all('TIME')
    for j in value_d : 
        value_data.append(j.text)
    for j in value_t :
        check_time = check_time_data(j.text)
        value_time.append(check_time)
    all_time.append(value_time)
    all_data.append(value_data)

all_time = [[convert_date_format(date) for date in row] for row in all_time]


# korea economic indicator table insert
korea_economic_indicator_df = date_df.copy()
for i in range(0, 11) :
    ko_eco_indi = pd.DataFrame({'Date' :  pd.to_datetime(all_time[i]), data_name[i] : all_data[i]}).set_index('Date')
    korea_economic_indicator_df = korea_economic_indicator_df.join(ko_eco_indi)
korea_economic_indicator_df = korea_economic_indicator_df.reset_index().rename(columns={'index' : 'fd_date'})

korea_economic_indicator_df.fillna(method='ffill', inplace=True)
korea_economic_indicator_df['fd_date'] = korea_economic_indicator_df['fd_date'].astype(str).map(lambda x : x[:10])


# ************************************** tb_us_economic_indicator **************************************
fred = Fred(api_key=fred_api_key)

indicators = {
    "uei_gdp": "GDP",
    "uei_fed_rate": "FEDFUNDS",
    "uei_ipi": "IR",
    "uei_ppi": "PPIACO",
    "uei_cpi": "CPIAUCSL",
    "uei_cpi_m": "CPIAUCNS",
    "uei_trade": "BOPGSTB",
    "uei_cb_cc": "CSCICP03USM665S",
    "uei_ps_m": "PCE",
    "uei_rs_m": "RSXFS",
    "uei_umich_cs": "UMCSENT"
}

us_economic_indicator_dic = {}
for name, series_id in indicators.items():
    try:
        us_economic_indicator_dic[name] = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    except ValueError as e:
        print(f"Error fetching {name}: {e}")

us_economic_indicator_df = date_df.copy()
wei_dic = pd.DataFrame(us_economic_indicator_dic)
us_economic_indicator_df = us_economic_indicator_df.join(wei_dic)
us_economic_indicator_df = us_economic_indicator_df.reset_index().rename(columns={'index' : 'fd_date'})
us_economic_indicator_df.fillna(method='ffill', inplace=True)
us_economic_indicator_df['fd_date'] = us_economic_indicator_df['fd_date'].astype(str).map(lambda x : x[:10])


# tb_finance_date
date_df = date_df.reset_index().rename(columns={'index' : 'fd_date'})
date_df['fd_date'] = date_df['fd_date'].astype(str).map(lambda x : x[:10])

# Excel save
df_len.append(len(stock_df.dropna(axis=0)))
df_len.append(len(main_economic_index_df.dropna(axis=0)))
df_len.append(len(korea_economic_indicator_df.dropna(axis=0)))
df_len.append(len(us_economic_indicator_df.dropna(axis=0)))
df_len = min(df_len)

# create dataframes of stock
df_finance_date = date_df.tail(df_len)
df_stock = stock_df.tail(df_len)
df_main_economic_index = main_economic_index_df.tail(df_len)
df_korea_economic_indicator = korea_economic_indicator_df.tail(df_len)
df_us_economic_indicator = us_economic_indicator_df.tail(df_len)


# 데이터프레임 순서 지정
all_tables_df = [
    df_finance_date, df_stock, df_korea_economic_indicator, 
    df_us_economic_indicator, df_main_economic_index, df_news, df_stock_predict
]

# 데이터베이스 연결
conn = pymysql.connect(
    host=database_host_ip,
    user=database_id,
    password=database_passwd,
    charset=database_charset
)

cur = conn.cursor()
print(f'========== DATABASE Connect ==========')

# 데이터베이스 사용 
cur.execute(f"USE {database_name}")


print('========== Stock Predict DELETE query start! ==========')
# 데이터가 있는지 확인하는 쿼리 실행
cur.execute('SELECT COUNT(*) FROM tb_stock_predict')
count = cur.fetchone()[0]  # 첫 번째 요소에서 카운트 값을 가져옴

if count > 0:
    print('========== Stock Predict DELETE query start! ==========')
    cur.execute('''
        DELETE FROM tb_stock_predict ORDER BY sp_id ASC LIMIT 1;
    ''')
    print('Delete executed successfully.')
else:
    print('No data to delete.')

print('========== Stock Predict DELETE query end ==========')


print(f'========== Insert query start! ==========')
# 데이터프레임을 데이터베이스에 삽입 또는 업데이트하는 함수
def insert_or_update_data(df, table_name, columns, values):
    print(f'========== {table_name} insert or update start ==========')
    for index, row in df.iterrows():
        # ON DUPLICATE KEY UPDATE를 사용하여 삽입 또는 업데이트
        update_stmt = ', '.join([f"{col} = VALUES({col})" for col in columns])  # UPDATE 부분
        cur.execute(f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({', '.join(['%s'] * len(values))})
            ON DUPLICATE KEY UPDATE {update_stmt};
        """, tuple(row[values]))
    print(f'========== {table_name} insert or update end ==========')

# 테이블 이름 리스트
database_tables_name = [
    'tb_finance_date', # 1
    'tb_stock', # 2
    'tb_korea_economic_indicator', # 3
    'tb_us_economic_indicator', # 4
    'tb_main_economic_index', # 5
    'tb_news', # 6
    'tb_stock_predict', # 7
]

# 각 테이블의 컬럼 정의
columns_finance_date = ['fd_date']
values_finance_date = ['fd_date']

columns_stock = ['fd_date', 'sc_ss_stock', 'sc_ss_per', 'sc_ss_pbr', 'sc_ss_roe', 'sc_ss_mc', 'sc_ap_stock', 'sc_ap_per', 'sc_ap_pbr', 'sc_ap_roe', 'sc_ap_mc', 'sc_coin']
values_stock = ['fd_date', 'sc_ss_stock', 'sc_ss_per', 'sc_ss_pbr', 'sc_ss_roe', 'sc_ss_mc', 'sc_ap_stock', 'sc_ap_per', 'sc_ap_pbr', 'sc_ap_roe', 'sc_ap_mc', 'sc_coin']

columns_korea_economic_indicator = ['fd_date', 'kei_gdp', 'kei_m2_end', 'kei_m2_avg', 'kei_fed_rate', 'kei_ppi', 'kei_ipi', 'kei_cpi', 'kei_imp', 'kei_exp', 'kei_cs', 'kei_bsi']
values_korea_economic_indicator = ['fd_date', 'kei_gdp', 'kei_m2_end', 'kei_m2_avg', 'kei_fed_rate', 'kei_ppi', 'kei_ipi', 'kei_cpi', 'kei_imp', 'kei_exp', 'kei_cs', 'kei_bsi']

columns_us_economic_indicator = ['fd_date', 'uei_gdp', 'uei_fed_rate', 'uei_ipi', 'uei_ppi', 'uei_cpi', 'uei_cpi_m', 'uei_trade', 'uei_cb_cc', 'uei_ps_m', 'uei_rs_m', 'uei_umich_cs']
values_us_economic_indicator = ['fd_date', 'uei_gdp', 'uei_fed_rate', 'uei_ipi', 'uei_ppi', 'uei_cpi', 'uei_cpi_m', 'uei_trade', 'uei_cb_cc', 'uei_ps_m', 'uei_rs_m', 'uei_umich_cs']

columns_main_economic_index = ['fd_date', 'mei_nasdaq', 'mei_sp500', 'mei_dow', 'mei_kospi', 'mei_gold', 'mei_oil', 'mei_ex_rate']
values_main_economic_index = ['fd_date', 'mei_nasdaq', 'mei_sp500', 'mei_dow', 'mei_kospi', 'mei_gold', 'mei_oil', 'mei_ex_rate']

columns_news = ['news_title', 'news_simple_text', 'news_link', 'news_classification']
values_news = ['news_title', 'news_simple_text', 'news_link', 'news_classification']

columns_stock_predict = ['sp_date', 'sp_ss_predict', 'sp_ap_predict', 'sp_bit_predict']
values_stock_predict = ['sp_date', 'sp_ss_predict', 'sp_ap_predict', 'sp_bit_predict']

column_tables = [columns_finance_date, columns_stock, columns_korea_economic_indicator, 
                columns_us_economic_indicator, columns_main_economic_index, columns_news, columns_stock_predict]

value_tables = [values_finance_date, values_stock, values_korea_economic_indicator, 
                values_us_economic_indicator, values_main_economic_index, values_news, values_stock_predict]

# 데이터프레임 삽입/업데이트
insert_or_update_data(all_tables_df[0], database_tables_name[0], column_tables[0], value_tables[0])
insert_or_update_data(all_tables_df[1], database_tables_name[1], column_tables[1], value_tables[1])
insert_or_update_data(all_tables_df[2], database_tables_name[2], column_tables[2], value_tables[2])
insert_or_update_data(all_tables_df[3], database_tables_name[3], column_tables[3], value_tables[3])
insert_or_update_data(all_tables_df[4], database_tables_name[4], column_tables[4], value_tables[4])
insert_or_update_data(all_tables_df[5], database_tables_name[5], column_tables[5], value_tables[5])
insert_or_update_data(all_tables_df[6], database_tables_name[6], column_tables[6], value_tables[6])

print('========== Update Query End ==========')

conn.commit()
conn.close()
print('========== DATABASE Connect End ==========')