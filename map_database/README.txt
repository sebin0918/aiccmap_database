## data_to_excel.ipynb
* API 및 Python Library를 이용한 데이터 수집 코드 파일입니다.
API 및 Python Library : yfinance, pykrx, finance-datareader, 한국은행 API, Fred API

## make_sample_data.ipynb
* 데이터베이스 적재 테스트용 데이터입니다.

## MAP_project_DATABASE.ipynb
* 데이터베이스 초기 조작 코드 파일입니다.
./datas 폴더에 저장되어 있는 table별 엑셀 파일의 데이터를 DATABASE에 저장합니다.
아래 MAP_prject_create_table.sql 파일과 같은 폴더에 위치시켜야 코드가 정상적으로 실행됩니다.

## MAP_prject_create_table.sql
* Database table을 만드는 create Query가 작성된 파일입니다.
위 MAP_project_DATABASE.ipynb 파일과 같은 폴더에 위치시켜야 위 코드가 정상적으로 실행됩니다.
위 코드에서 해당 sql 파일을 실행시키므로 따로 실행이 필요 없습니다.

## table_list.txt
* 작성될 DATABASE의 table 이름이 작성되어 있습니다.

## MAP(My Asset Plan)_DATABASE_ERD.png
* DATABASE의 ERD입니다.

# 파일 실행 순서
make_sample_data.ipynb => data_to_excel.ipynb => MAP_project_DATABASE.ipynb