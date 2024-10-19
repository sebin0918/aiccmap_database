# 데이터 처리 및 분석 관련 라이브러리
import faiss
import numpy as np
import pandas as pd
import networkx as nx
import random
import string

# 웹 관련 라이브러리
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# 자연어 처리 및 텍스트 분석 라이브러리
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('stopwords')

# Transformer 기반 언어 모델 관련 라이브러리
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, AutoModelForSequenceClassification, AutoTokenizer, BertModel, BertTokenizer

# 시간 및 정규 표현식 관련 라이브러리
from datetime import datetime, timedelta
import re
import time
import json

# 경고 및 로그 관리
import warnings
import logging
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
logging.getLogger("faiss").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

# 외부 라이브러리
from fredapi import Fred
from pykrx import stock as stk
import yfinance as yf
import requests
import pymysql
import pickle
import joblib
import torch.nn as nn

# 안전한 텐서 사용
from safetensors.torch import safe_open

# Faker 라이브러리
import faker

# 로깅 및 경고
import pytz

kst = pytz.timezone('Asia/Seoul')

# Faker 라이브러리 사용
fake = faker.Faker('ko_KR')  # 한국어 로케일 설정

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

# 난수 생성 함수
def generate_random_password(length=12):
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()"
    return ''.join(random.choice(characters) for _ in range(length))

def generate_random_bank_number(length=12):
    return ''.join(random.choices(string.digits, k=length))

def generate_phone_number():
    return '010' + ''.join(random.choices(string.digits, k=6))

# 금액을 00 단위로 맞추는 함수
def round_to_hundred(amount):
    return (amount // 100) * 100

# 데이터 생성 함수
def create_user_data(num_users=10, income_range=(2000, 5000), fixed_income_range=(2500000, 4500000),
                    expense_range=(5000, 20000), fixed_expense_range=(100000, 400000)):
    # tb_user_key (사용자 정보)
    user_ids = list(range(1, num_users + 1))  # user_id를 숫자 형태로 생성
    emails = [fake.email() for _ in range(num_users)]
    passwords = [generate_random_password() for _ in range(num_users)]
    permissions = [1] * (num_users)

    user_ids[0], emails[0], passwords[0], permissions[0] = 1, 'root@root.com', '1', 0 # 관리자 계정
    user_ids[1], emails[1], passwords[1], permissions[1] = 2, 'user@user.com', '1', 0
    user_ids[2], emails[2], passwords[2], permissions[2] = 3, 'test@test.com', '1', 1

    df_user_key = pd.DataFrame({
        'user_id': user_ids,
        'uk_email': emails,
        'uk_password': passwords,
        'uk_permission': permissions
    })

    # tb_user_information (사용자 상세 정보)
    names = [fake.name() for _ in range(num_users)]
    birth_dates = [fake.date_of_birth(minimum_age=18, maximum_age=80) for _ in range(num_users)]
    sexes = [random.choice([0, 1]) for _ in range(num_users)]  # 성별을 랜덤으로 선택
    bank_nums = [generate_random_bank_number() for _ in range(num_users)]
    phone_numbers = [generate_phone_number() for _ in range(num_users)]

    df_user_information = pd.DataFrame({
        'user_id': user_ids,
        'ui_name': names,
        'ui_birth_date': birth_dates,
        'ui_sex': sexes,
        'ui_bank_num': bank_nums,
        'ui_caution': [0] * num_users,
        'ui_phone_number': phone_numbers
    })

    # tb_user_finance (사용자 재무 정보)
    capitals = np.random.randint(5000000, 50000000, size=num_users)  # 자본금
    loans = np.random.randint(0, 30000000, size=num_users)  # 대출
    installment_savings = np.random.randint(0, 20000000, size=num_users)  # 적금
    deposits = np.random.randint(0, 40000000, size=num_users)  # 예금
    target_budgets = np.random.randint(2000000, 8000000, size=num_users)  # 목표 예산

    df_user_finance = pd.DataFrame({
        'user_id': user_ids,
        'uf_capital': capitals,
        'uf_loan': loans,
        'uf_installment_saving': installment_savings,
        'uf_deposit': deposits,
        'uf_target_budget': target_budgets
    })

    # tb_received_paid (입출금 내역)
    details = {
        'income': ['선물', '임대', '보너스', '입금', '송금', '기타'],
        'expense': ['마트', '편의점', '식사', '출금', '송금']
    }
    fixed_incomes = ['월급', '이자', '배당금']
    fixed_expenses = ['보험료', '대출 상환', '공과금', '교통', '정기 예금', '적금', '통신비']

    date_a = '2024-01-01'
    start_date = datetime(int(date_a[:4]), int(date_a[5:7]), int(date_a[8:]))
    end_date = datetime.today()

    records = []
    
    # 각 사용자별 고정 수입 및 지출 값을 처음에 생성해 저장해둠
    fixed_income_values = {user_id: {income: round_to_hundred(random.randint(*fixed_income_range)) for income in fixed_incomes} for user_id in user_ids}
    fixed_expense_values = {user_id: {expense: round_to_hundred(random.randint(*fixed_expense_range)) for expense in fixed_expenses} for user_id in user_ids}

    for user_id, loan in zip(user_ids, df_user_finance['uf_loan']):
        current_date = start_date
        last_fixed_income_month = -1
        last_fixed_expense_month = -1
        loan_payment_done = False
        while current_date <= end_date:
            if current_date.day in [15, 25]:
                # 고정 수입 내역 (매달 같은 값)
                for detail in fixed_incomes:
                    if current_date.month != last_fixed_income_month:
                        amount = fixed_income_values[user_id][detail]
                        rp_hold = 0
                        records.append([user_id, current_date, detail, amount, rp_hold, 0])
                        last_fixed_income_month = current_date.month
                        
                # 고정 지출 내역 (매달 같은 값, 대출 상환은 한 달에 한 번만 기록)
                if current_date.month != last_fixed_expense_month:
                    for detail in fixed_expenses:
                        if detail == '대출 상환' and loan > 0 and not loan_payment_done:
                            amount = min(loan, fixed_expense_values[user_id][detail])
                            loan -= amount
                            rp_hold = 0
                            records.append([user_id, current_date, detail, amount, rp_hold, 1])
                            loan_payment_done = True
                        elif detail != '대출 상환':
                            amount = fixed_expense_values[user_id][detail]
                            rp_hold = 0
                            records.append([user_id, current_date, detail, amount, rp_hold, 1])
                    last_fixed_expense_month = current_date.month
                    loan_payment_done = False  # 다음 달에 다시 상환 가능

            # 기타 수입 및 지출 내역 (랜덤하게 하루 최대 2개 생성)
            for detail in details['income']:
                if random.random() < 0.2:
                    amount = round_to_hundred(random.randint(*income_range))
                    rp_hold = 1
                    records.append([user_id, current_date, detail, amount, rp_hold, 0])

            for detail in details['expense']:
                if random.random() < 0.2 and detail != '대출 상환':
                    amount = round_to_hundred(random.randint(*expense_range))
                    rp_hold = 1
                    records.append([user_id, current_date, detail, amount, rp_hold, 1])

            current_date += timedelta(days=1)

    df_received_paid = pd.DataFrame(records, columns=['user_id', 'rp_date', 'rp_detail', 'rp_amount', 'rp_hold', 'rp_part'])

    # tb_shares_held (주식 거래 내역)
    shares_dates = pd.date_range(start=date_a, end=datetime.today(), freq='D').to_list()

    share_records = []
    for user_id in user_ids:
        for date in shares_dates:
            # 주식 매수/매도 내역을 -10에서 10까지 랜덤으로 설정
            ss_count = np.random.randint(-10, 11)  # 삼성 주식 매수/매도
            ap_count = np.random.randint(-10, 11)  # 애플 주식 매수/매도
            bit_count = np.random.randint(-10, 11)  # 비트코인 매수/매도

            share_records.append([user_id, date, ss_count, ap_count, bit_count])

    df_shares_held = pd.DataFrame(share_records, columns=['user_id', 'sh_date', 'sh_ss_count', 'sh_ap_count', 'sh_bit_count'])

    return df_user_key, df_user_information, df_user_finance, df_received_paid, df_shares_held

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


def searchNaverNews_stock_market(query, display=10, start=1):
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
        filtered_news = searchNaverNews_stock_market(query, display=display)
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
# 구분 값 !eo$


# ================================================================== stock data ==================================================================

print('============= stock data API Start =============')
print('Start time : ', datetime.today())

# 날짜 형식 2024-9-6
start_day = '2014-01-01'
month_date = str(datetime.now().month)
day_date = str(datetime.now().day)
if len(month_date) == 1:
    month_date = f'0{month_date}'
if len(day_date) == 1:
    day_date = f'0{day_date}'
end_day = f'{datetime.now().year}-{month_date}-{day_date}'  # '2024-08-28'

start_date = datetime.strptime(start_day, '%Y-%m-%d')
end_date = datetime.strptime(end_day, '%Y-%m-%d')

date_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))

df_len = []  # 데이터 길이를 조절하기 위한 리스트

# ************************************** tb_stock **************************************
# 데이터를 가져오기 위한 날짜 범위 설정

# Samsung(005930)
ticker = '005930.KS'
samsung = yf.Ticker(ticker)
samsung_stock = samsung.history(start=start_date, end=end_date)

samsung_stock.rename(columns={'Close' : 'sc_ss_stock'}, inplace=True)
samsung_stock = samsung_stock['sc_ss_stock']
samsung_stock = samsung_stock.reset_index()
samsung_stock['Date'] = samsung_stock['Date'].astype(str).map(lambda x : x[:10])
samsung_stock['Date'] = pd.to_datetime(samsung_stock['Date'])
samsung_stock.set_index(keys='Date', inplace=True)

# Samsung Market Capitalization
samsung_data = yf.download(ticker, start=start_date, end=end_date)
close_prices = samsung_data['Close']
shares_outstanding = samsung.info['sharesOutstanding']
samsung_market = close_prices * shares_outstanding
samsung_Market_Cap = pd.DataFrame({'sc_ss_mc': samsung_market})
samsung_Market_Cap['sc_ss_mc'] = samsung_Market_Cap['sc_ss_mc'].map(lambda x : int(x))

# Samsung PER, PBR, ROE
samsung_PER_PBR_ROE = stk.get_market_fundamental(start_day, end_day, "005930")[['PER', 'PBR']] # 삼성전자
samsung_PER_PBR_ROE = stk.get_market_fundamental(start_day, end_day, "005930")[['PER', 'PBR']] # 삼성전자
samsung_PER_PBR_ROE.rename(columns={'PER' : 'sc_ss_per', 'PBR' : 'sc_ss_pbr'}, inplace=True)
samsung_PER_PBR_ROE['sc_ss_roe'] = samsung_PER_PBR_ROE['sc_ss_pbr'] / samsung_PER_PBR_ROE['sc_ss_per']

# Apple(AAPL)
# Apple Stock
ticker = 'AAPL'
apple = yf.Ticker(ticker)
apple_stock = apple.history(start=start_date, end=end_date)

apple_stock.rename(columns={'Close' : 'sc_ap_stock'}, inplace=True)
apple_stock = apple_stock['sc_ap_stock']
apple_stock = apple_stock.reset_index()
apple_stock['Date'] = apple_stock['Date'].astype(str).map(lambda x : x[:10])
apple_stock['Date'] = pd.to_datetime(apple_stock['Date'])
apple_stock.set_index(keys='Date', inplace=True)

# Apple Market Capitalization
apple_data = apple.history(start=start_date, end=end_date)
apple_Market_Cap = apple.info['sharesOutstanding'] * apple_data['Close']
apple_Market_Cap = pd.DataFrame({'sc_ap_mc': apple_Market_Cap})
apple_Market_Cap = apple_Market_Cap.reset_index()
apple_Market_Cap['Date'] = apple_Market_Cap['Date'].astype(str).map(lambda x : x[:10])
apple_Market_Cap['Date'] = pd.to_datetime(apple_Market_Cap['Date'])
apple_Market_Cap.set_index(keys='Date', inplace=True)
apple_Market_Cap['sc_ap_mc'] = apple_Market_Cap['sc_ap_mc'].map(lambda x : int(x))

# Apple PER, PBR, ROE
apple_PER_PBR_ROE = get_per_pbr_df("AAPL", start_day, end_day)
apple_PER_PBR_ROE.rename(columns={'PER' : 'sc_ap_per', 'PBR' : 'sc_ap_pbr'}, inplace=True)
apple_PER_PBR_ROE = apple_PER_PBR_ROE.reset_index(drop=False)
apple_PER_PBR_ROE['Date'] = apple_PER_PBR_ROE['Date'].astype(str).map(lambda x : x[:10])
apple_PER_PBR_ROE['Date'] = pd.to_datetime(apple_PER_PBR_ROE['Date'])
apple_PER_PBR_ROE.set_index(keys='Date', inplace=True)
apple_PER_PBR_ROE['sc_ap_roe'] = apple_PER_PBR_ROE['sc_ap_pbr'].astype(float) / apple_PER_PBR_ROE['sc_ap_per'].astype(float)



# Bit-Coin(BTC)
ticker = 'BTC-USD'
bitcoin = yf.Ticker(ticker)
bitcoin_stock = bitcoin.history(start=start_date, end=end_date)

bitcoin_stock.rename(columns={'Close' : 'sc_coin'}, inplace=True)
bitcoin_stock = bitcoin_stock['sc_coin']

bitcoin_stock = bitcoin_stock.reset_index()
bitcoin_stock['Date'] = bitcoin_stock['Date'].astype(str).map(lambda x : x[:10])
bitcoin_stock['Date'] = pd.to_datetime(bitcoin_stock['Date'])
bitcoin_stock.set_index(keys='Date', inplace=True)

# stock table insert
tb_stock_df_list = [samsung_stock, samsung_PER_PBR_ROE, samsung_Market_Cap,
                    apple_stock, apple_PER_PBR_ROE, apple_Market_Cap, 
                    bitcoin_stock]

stock_df = date_df.copy()
for i in range(len(tb_stock_df_list)) :
    stock_df = stock_df.join(tb_stock_df_list[i])

stock_df = stock_df.reset_index().rename(columns={'index' : 'fd_date'})
stock_df.fillna(method='ffill', inplace=True)
stock_df['fd_date'] = stock_df['fd_date'].astype(str).map(lambda x : x[:10])

stock_df['fd_date'] = pd.to_datetime(stock_df['fd_date'])
print(stock_df)

# XGBoost 예측 함수
def xgboost_predict(stock, df, model_filename):
    stock_dataset = df[['fd_date', stock]].set_index('fd_date')
    start_day = 300
    forecast_days = 14 # 예측 날짜 변경

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
def sarima_predict_up_to_date(stock, model_path, stock_df, lookback_days=60, steps_ahead=14): # steps_ahead 예측날짜 변경
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
nasdaq = yf.download('^IXIC', start='2014-01-01', end='2024-12-31')
nasdaq = yf.download('^IXIC', start='2014-01-01', end='2024-12-31')
nasdaq.rename(columns={'Close': 'mei_nasdaq'}, inplace=True)
nasdaq = nasdaq['mei_nasdaq']

# S&P 500
snp500 = yf.download('^GSPC', start='2014-01-01', end='2024-12-31')
snp500 = yf.download('^GSPC', start='2014-01-01', end='2024-12-31')
snp500.rename(columns={'Close': 'mei_sp500'}, inplace=True)
snp500 = snp500['mei_sp500']

# Dow Jones Industrial Average (DJI)
dow = yf.download('^DJI', start='2014-01-01', end='2024-12-31')
dow = yf.download('^DJI', start='2014-01-01', end='2024-12-31')
dow.rename(columns={'Close': 'mei_dow'}, inplace=True)
dow = dow['mei_dow']

# KOSPI
kospi = yf.download('^KS11', start='2014-01-01', end='2024-12-31')
kospi = yf.download('^KS11', start='2014-01-01', end='2024-12-31')
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
dollar_to_won = yf.download('KRW=X', '2014-01-01')
dollar_to_won = yf.download('KRW=X', '2014-01-01')
dollar_to_won.rename(columns={'Close' : 'mei_ex_rate'}, inplace=True)
dollar_to_won = dollar_to_won['mei_ex_rate']

# main economic index table insert
tb_main_economic_index_df_list = [nasdaq, snp500, dow, kospi, gold_data, oil_data, dollar_to_won]

main_economic_index_df = date_df.copy()
for i in range(len(tb_main_economic_index_df_list)) :
    main_economic_index_df = main_economic_index_df.join(tb_main_economic_index_df_list[i])

main_economic_index_df = main_economic_index_df.reset_index().rename(columns={'index' : 'fd_date'})
main_economic_index_df.fillna(method='ffill', inplace=True)
main_economic_index_df['fd_date'] = main_economic_index_df['fd_date'].astype(str).map(lambda x : x[:10]) #'2024-09-06 00:00:00'


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

api_link = ['/200Y102/Q/2014Q1/2024Q2/10111',
            '/101Y007/M/201401/202406/BBIA00',
            '/101Y008/M/201401/202406/BBJA00',
            '/722Y001/M/201401/202406/0101000',
            '/404Y014/M/201401/202406/*AA',
            '/401Y015/M/201401/202406/*AA/W',
            '/901Y009/M/201401/202406/0',
            '/403Y003/M/201401/202406/*AA',
            '/403Y001/M/201401/202406/*AA',
            '/511Y002/M/201401/202406/FME/99988',
            '/512Y007/M/201401/202406/AA/99988']

all_data = []
all_time = []
for i in range(len(api_link)) :
    value_time = []
    value_data = []
    search_url = f'https://ecos.bok.or.kr/api/StatisticSearch/{ko_bank_key}/xml/kr/1/1{api_link[i]}'

    search_respons = requests.get(search_url)
    search_xml = search_respons.text
    search_soup = BeautifulSoup(search_xml, 'xml')
    total_val = search_soup.find('list_total_count')

    url = f'https://ecos.bok.or.kr/api/StatisticSearch/{ko_bank_key}/xml/kr/1/{total_val.text}{api_link[i]}'
    respons = requests.get(url)
    title_xml = respons.text
    title_soup = BeautifulSoup(title_xml, 'xml') 
    value_d = title_soup.find_all('DATA_VALUE')
    value_t = title_soup.find_all('TIME')# '<TIME>20240906</TIME>'
    for j in value_d : 
        value_data.append(j.text)
    for j in value_t :
        check_time = check_time_data(j.text) # 20240906
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


df_user_key, df_user_information, df_user_finance, df_received_paid, df_shares_held = create_user_data()
df_user_key, df_user_information, df_user_finance, df_received_paid, df_shares_held = create_user_data()

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



# 데이터베이스 연결 정보
conn = pymysql.connect(
    host=database_host_ip,
    user=database_id,
    password=database_passwd,
    charset=database_charset
)

cur = conn.cursor()
print(f'========== DATABASE Connect ==========')

# 데이터베이스 생성 및 SQL 파일 실행
cur.execute(f"DROP DATABASE IF EXISTS {database_name}")
cur.execute(f"CREATE DATABASE {database_name}")
cur.execute(f"USE {database_name}")

with open(sql_file_path, 'r', encoding='utf-8') as sql_file:
    sql_commands = sql_file.read().split(';')
    for command in sql_commands:
        if command.strip():
            cur.execute(command)

print(f'========== Insert query start! ==========')

# 데이터프레임을 데이터베이스에 삽입하는 함수
def insert_data(df, table_name, columns, values):
    print(f'========== {table_name} insert start ==========')
    for index, row in df.iterrows():
        cur.execute(f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES ({', '.join(['%s'] * len(values))});
        """, tuple(row[values]))
    print(f'========== {table_name} insert end ==========')

# 테이블별 데이터 삽입
table_names = {
    0: 'tb_user_key',
    1: 'tb_user_information',
    2: 'tb_user_finance',
    3: 'tb_received_paid',
    4: 'tb_shares_held',
    5: 'tb_finance_date',
    6: 'tb_stock',
    7: 'tb_korea_economic_indicator',
    8: 'tb_us_economic_indicator',
    9: 'tb_main_economic_index',
    10: 'tb_news',
    11: 'tb_stock_predict',
}

columns_user_key = ['uk_email', 'uk_password', 'uk_permission']
values_user_key = ['uk_email', 'uk_password', 'uk_permission']

columns_user_information = ['user_id', 'ui_name', 'ui_birth_date', 'ui_sex', 'ui_bank_num', 'ui_caution', 'ui_phone_number']
values_user_information = ['user_id', 'ui_name', 'ui_birth_date', 'ui_sex', 'ui_bank_num', 'ui_caution', 'ui_phone_number']

columns_user_finance = ['user_id', 'uf_capital', 'uf_loan', 'uf_installment_saving', 'uf_deposit', 'uf_target_budget']
values_user_finance = ['user_id', 'uf_capital', 'uf_loan', 'uf_installment_saving', 'uf_deposit', 'uf_target_budget']

columns_received_paid = ['user_id', 'rp_date', 'rp_detail', 'rp_amount', 'rp_hold', 'rp_part']
values_received_paid = ['user_id', 'rp_date', 'rp_detail', 'rp_amount', 'rp_hold', 'rp_part']

columns_shares_held = ['user_id', 'sh_date', 'sh_ss_count', 'sh_ap_count', 'sh_bit_count']
values_shares_held = ['user_id', 'sh_date', 'sh_ss_count', 'sh_ap_count', 'sh_bit_count']

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

# 데이터 삽입
insert_data(df_user_key, table_names[0], columns_user_key, values_user_key)
insert_data(df_user_information, table_names[1], columns_user_information, values_user_information)
insert_data(df_user_finance, table_names[2], columns_user_finance, values_user_finance)
insert_data(df_received_paid, table_names[3], columns_received_paid, values_received_paid)
insert_data(df_shares_held, table_names[4], columns_shares_held, values_shares_held)
insert_data(df_finance_date, table_names[5], columns_finance_date, values_finance_date)
insert_data(df_stock, table_names[6], columns_stock, values_stock)
insert_data(df_korea_economic_indicator, table_names[7], columns_korea_economic_indicator, values_korea_economic_indicator)
insert_data(df_us_economic_indicator, table_names[8], columns_us_economic_indicator, values_us_economic_indicator)
insert_data(df_main_economic_index, table_names[9], columns_main_economic_index, values_main_economic_index)
insert_data(df_news, table_names[10], columns_news, values_news)
insert_data(df_stock_predict, table_names[11], columns_stock_predict, values_stock_predict)


# 추가적인 테이블 삽입 예시 (나머지 테이블도 동일한 형식으로 추가)
# 예: insert_data(df_finance_date, table_names[5], columns_finance_date, values_finance_date)
# 필요한 데이터프레임 생성 후 insert_data 함수를 사용하여 삽입하세요.

print('========== Insert Query End ==========')

conn.commit()
conn.close()
print('========== DATABASE Connect End ==========')
