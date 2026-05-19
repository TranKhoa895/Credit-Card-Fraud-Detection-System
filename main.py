import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import time
'''
B1: ETL-Trích xuất và Lưu trữ Dữ liệu (Extract, Transform, Load)
'''
# 1. Kết nối SQL Server
SERVER = r'PHUOCKHOA-P14S\SQLEXPRESS'
DATABASE = 'fraud_detection'

connection_string = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={SERVER};'
    f'DATABASE={DATABASE};'
    f'Trusted_Connection=yes;'
)

# 2. Tạo kết nối
try: 
    quoted_conn = urllib.parse.quote_plus(connection_string)
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={quoted_conn}',
                           fast_executemany=True)
    
    print(f'Đang kết nối tới Server: {SERVER}')
    print("Bắt đầu quy trình nạp dữ liệu vào SQL")
    start_time = time.time()

# 3. Đọc và đẩy dữ liệu
# File Train
    print("Đang đọc và nạp file 'fraudTrain.csv' vào SQL Server")
    
    chunk_count_train = 0
    for chunk in pd.read_csv('fraudTrain.csv', chunksize = 50000):
        if chunk_count_train == 0:
            chunk.to_sql('fraud_train', con=engine, if_exists='replace', index=False)
        else:
            chunk.to_sql('fraud_train', con=engine, if_exists='append', index=False)

        chunk_count_train += 1
        print(f'Đã nạp xong {chunk_count_train*50000} dòng của tập TRAIN!')    
# File Test
    print("\nĐang đọc và nạp file 'fraudTest.csv' vào SQL Server")
    
    chunk_count_test = 0
    for chunk in pd.read_csv('fraudTest.csv', chunksize = 50000):
        if chunk_count_test == 0:
            chunk.to_sql('fraud_test', con=engine, if_exists='replace', index=False)
        else:
            chunk.to_sql('fraud_test', con=engine, if_exists='append', index=False)

        chunk_count_test += 1
        print(f'Đã nạp xong {chunk_count_test*50000} dòng của tập TEST!') 

    end_time = time.time()
    print("\nHoàn thành xong phần ETL!")
    print(f'Tổng thời gian nạp dữ liệu: {(end_time-start_time)/60:.4f} phút')
except Exception as e:
    print(f'Lỗi trong quá trình nạp dữ liệu: {e}')


