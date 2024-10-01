from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from fredapi import Fred
from pykrx import stock
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import pymysql
import random
import string
import faker


# ================================================================== Function ==================================================================

# Faker 라이브러리 사용
fake = faker.Faker('ko_KR')  # 한국어 로케일 설정

# 데이터베이스 연결 정보
database_info = open('./database_id.txt', 'r').readlines()

sql_file_path = database_info[0].replace('\n', '')
database_host_ip = database_info[1].replace('\n', '')
database_name = database_info[2].replace('\n', '')
database_id = database_info[3].replace('\n', '')
database_passwd = database_info[4].replace('\n', '')
database_charset = database_info[5].replace('\n', '')

# API KEY
API_key = open('./api_key.txt', 'r').readlines()

ko_bank_key = API_key[0].replace('\n', '')
fred_api_key = API_key[1].replace('\n', '')

# 난수 생성 함수
def generate_random_password(length=12):
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()"
    return ''.join(random.choice(characters) for _ in range(length))

def generate_random_bank_number(length=12):
    return ''.join(random.choices(string.digits, k=length))

def generate_phone_number():
    return '010' + ''.join(random.choices(string.digits, k=6))

# 데이터 생성 함수
def create_user_data(num_users=10):
    # tb_user_key
    user_ids = list(range(1, num_users + 1))  # user_id를 숫자 형태로 생성
    emails = [fake.email() for _ in range(num_users)]
    passwords = [generate_random_password() for _ in range(num_users)]
    permissions = [0] + [1] * (num_users - 1)

    user_ids[0], emails[0], passwords[0], permissions[0] = 1, 'root@root.com', '1', 0

    df_user_key = pd.DataFrame({
        'user_id': user_ids,
        'uk_email': emails,
        'uk_password': passwords,
        'uk_permission': permissions
    })

    # tb_user_information
    names = [fake.name() for _ in range(num_users)]
    birth_dates = [fake.date_of_birth(minimum_age=18, maximum_age=80) for _ in range(num_users)]
    sexes = [0] + [1] * (num_users - 1)
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

    # tb_user_finance
    capitals = np.random.randint(1000, 100000, size=num_users)
    loans = np.random.randint(0, 50000, size=num_users)
    installment_savings = np.random.randint(0, 20000, size=num_users)
    deposits = np.random.randint(0, 100000, size=num_users)
    target_budgets = np.random.randint(10000, 50000, size=num_users)

    df_user_finance = pd.DataFrame({
        'user_id': user_ids,
        'uf_capital': capitals,
        'uf_loan': loans,
        'uf_installment_saving': installment_savings,
        'uf_deposit': deposits,
        'uf_target_budget': target_budgets
    })

    # tb_received_paid
    details = {
        'income': ['Salary', 'Gift', 'Interest', 'Rental', 'Bonus', 'Refund', 'Others'],
        'expense': ['Rent', 'Groceries', 'Utilities', 'Transport', 'Dining', 'Entertainment', 'Healthcare']
    }
    fixed_incomes = ['Salary', 'Bonus', 'Refund']
    fixed_expenses = ['Rent', 'Utilities', 'Transport']

    records = []
    for user_id in user_ids:
        start_date = datetime(2014, 1, 1)
        end_date = datetime.now()
        current_date = start_date

        while current_date <= end_date:
            # 고정 수입 (매월 1일)
            if current_date.day == 1:
                for income in fixed_incomes:
                    amount = random.randint(5000, 10000)
                    records.append([user_id, current_date, income, amount, 0, 0])

            # 고정 지출 (매월 5일)
            if current_date.day == 5:
                for expense in fixed_expenses:
                    amount = random.randint(1000, 5000)
                    records.append([user_id, current_date, expense, amount, 0, 1])

            # 랜덤 입출금 내역 (하루 최대 5개)
            num_transactions = random.randint(1, 5)
            for _ in range(num_transactions):
                transaction_type = random.choice(['income', 'expense'])
                detail = random.choice(details[transaction_type])
                amount = random.randint(1000, 5000) if transaction_type == 'income' else random.randint(500, 2000)
                part = 0 if transaction_type == 'income' else 1
                records.append([user_id, current_date, detail, amount, 0, part])

            # 다음 날로 이동
            current_date += timedelta(days=1)

    df_received_paid = pd.DataFrame(records, columns=['user_id', 'rp_date', 'rp_detail', 'rp_amount', 'rp_hold', 'rp_part'])

    # tb_shares_held (주식 보유 내역)
    share_records = []
    start_date = datetime(2014, 1, 1)
    end_date = datetime.now()

    for user_id in user_ids:
        current_date = start_date
        ss_count = 0
        ap_count = 0
        bit_count = 0

        while current_date <= end_date:
            # 매일 주식을 매수/매도할 확률 (50% 확률로 변화)
            if random.random() > 0.5:
                ss_count += random.randint(-5, 10)  # 삼성전자 주식 변동
                ap_count += random.randint(-5, 10)  # 애플 주식 변동
                bit_count += random.randint(-5, 10)  # 비트코인 변동

                # 음수가 되지 않도록 주식 개수 조정
                ss_count = max(ss_count, 0)
                ap_count = max(ap_count, 0)
                bit_count = max(bit_count, 0)

                # 날짜별 보유 주식 기록
                share_records.append([user_id, current_date, ss_count, ap_count, bit_count])

            current_date += timedelta(days=1)

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

# ================================================================== User data ==================================================================


df_user_key, df_user_information, df_user_finance, df_received_paid, df_shares_held = create_user_data()

# tb_news
tb_news_data = {
    'news_id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'news_title': ['호재 1', '호재 2', '악재 3', '악재 4', '호재 5', '호재 6', '악재 7', '악재 8', '악재 9'],
    'news_simple_text': ['호재', '호재호재', '악재악재악재', '악재악재악재악재', '호재호재호재호재호재', '호재호재호재호재호재호재', '악재악재악재악재악재악재악재', '악재악재악재악재악재악재악재악재', '악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재악재'],
    'news_link': ['http://example.com/1', 'http://example.com/2', 'http://example.com/3', 'http://example.com/4', 'http://example.com/5', 'http://example.com/2', 'http://example.com/3', 'http://example.com/4', 'http://example.com/5'],
    'news_classification' : [0, 0, 1, 1, 0, 0, 1, 1, 1]
}

df_news = pd.DataFrame(tb_news_data)

# ================================================================== stock data ==================================================================

print('============= stock data API Start =============')
print('Start time : ', datetime.today())

# 날짜 형식 2024-9-6
start_day = '2014-01-01'
month_date = str(datetime.now().month)
day_date = str(datetime.now().day)
if len(month_date) == 1 :
    month_date = f'0{month_date}'
if len(day_date) == 1 :
    day_date = f'0{day_date}'
end_day = f'{datetime.now().year}-{month_date}-{day_date}' #'2024-08-28'

start_date = datetime(int(start_day[:4]), int(start_day[5:7]), int(start_day[8:10]))
end_date = datetime(int(end_day[:4]), int(end_day[5:7]), int(end_day[8:10]))

date_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
df_len = [] # 데이터 길이를 조절하기 위한 리스트


# ************************************** tb_stock **************************************
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
samsung_PER_PBR_ROE = stock.get_market_fundamental(start_day, end_day, "005930")[['PER', 'PBR']] # 삼성전자
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


# 추가적인 테이블 삽입 예시 (나머지 테이블도 동일한 형식으로 추가)
# 예: insert_data(df_finance_date, table_names[5], columns_finance_date, values_finance_date)
# 필요한 데이터프레임 생성 후 insert_data 함수를 사용하여 삽입하세요.

print('========== Insert Query End ==========')

conn.commit()
conn.close()
print('========== DATABASE Connect End ==========')
