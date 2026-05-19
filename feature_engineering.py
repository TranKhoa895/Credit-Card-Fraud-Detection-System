import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import time
import os
'''
B3: FEATURE ENGINEERING 
'''
star_time = time.time()

# 1. Thiết lập kết nối trực tiếp với SQL Server
SERVER = r'PHUOCKHOA-P14S\SQLEXPRESS'
DATABASE = 'fraud_detection'

connection_string = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={SERVER};'
    f'DATABASE={DATABASE};'
    f'Trusted_Connection=yes;'
)
quoted_conn = urllib.parse.quote_plus(connection_string)
engine = create_engine(f'mssql+pyodbc:///?odbc_connect={quoted_conn}')

categories_list = pd.read_sql("SELECT DISTINCT category FROM fraud_train;", con=engine)['category'].tolist()

OUTPUT_FILE = "fraud_processed_full.csv"
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

query = """
SELECT amt, trans_date_trans_time, category, dob, is_fraud 
FROM fraud_train;
"""
chunk_idx = 1
total_rows = 0

try:
    for chunk in pd.read_sql(query, con=engine, chunksize=200000):
        if chunk.empty:
            break
        
# 2. Xử lý các đặc trưng hành vi
        chunk['trans_date_trans_time'] = pd.to_datetime(chunk['trans_date_trans_time'])
        chunk['dob'] = pd.to_datetime(chunk['dob'])

# 2.1. Khởi tạo các cột tính năng mới
        chunk['age'] = chunk['trans_date_trans_time'].dt.year - chunk['dob'].dt.year
        chunk['hour'] = chunk['trans_date_trans_time'].dt.hour
        chunk['is_night'] = chunk['hour'].apply(lambda x: 1 if x >= 23 or x <= 3 else 0)
        chunk['is_weekend'] = chunk['trans_date_trans_time'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)

# 2.2. Tính toán tỷ lệ chi tiêu bất thường theo từng danh mục
        category_avg = chunk.groupby('category')['amt'].transform('mean')
        chunk['atm_to_cat_avg_ratio'] = chunk['amt'] / category_avg

# 3. Mã hóa biến chữ (category) thành biến số số học (One-Hot Encoding)
        chunk['category'] = pd.Categorical(chunk['category'], categories=categories_list)
        chunk_encoded = pd.get_dummies(chunk, columns=['category'], drop_first=True)
        category_cols = [col for col in chunk_encoded.columns if 'category_' in col]
        chunk_encoded[category_cols] = chunk_encoded[category_cols].astype(int)

# 4. Loại bỏ các cột thô ban đầu để tối ưu không gian dữ liệu đưa vào AI
        columns_to_drop = ['trans_date_trans_time', 'dob', 'hour']
        chunk_final = chunk_encoded.drop(columns=columns_to_drop)

# 4.1. Đồng bộ toàn bộ các cột định dạng Boolean thành số nguyên 0/1
        bool_cols = chunk_final.select_dtypes(include=['bool']).columns
        chunk_final[bool_cols] = chunk_final[bool_cols].astype(int)

# 4.2. Ghi cuốn chiếu dữ liệu thành phẩm xuống ổ đĩa cứng
        if chunk_idx == 1:
            chunk_final.to_csv(OUTPUT_FILE, index=False)
        else:
            chunk_final.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)

        total_rows += len(chunk_final)
        chunk_idx += 1

# 5. Nghiệm thu kết quả
    end_time = time.time()
    print(f"Tổng số dòng đã xử lý thành công: {total_rows:,} dòng.")
    print(f"File kết quả sạch đã lưu trữ tại: {OUTPUT_FILE}")
    print(f"Tổng thời gian chạy toàn bộ hệ thống: {end_time - star_time:.2f} giây.")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 5)
    pd.set_option('display.width', 1000)

    if os.path.exists(OUTPUT_FILE):
        df_check = pd.read_csv(OUTPUT_FILE, nrows=20)
        print(df_check)

except Exception as e:
    print(f"Có lỗi xảy ra trong quá trình xử lý: {e}")