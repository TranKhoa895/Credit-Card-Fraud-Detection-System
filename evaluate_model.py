import pandas as pd
import time
import os
from sqlalchemy import create_engine
import urllib.parse
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib
'''
B5: EVALUATING MODEL (TEST DATA)
'''
start_time = time.time()

# 1. Tải AI đã đóng gói ở B4
MODEL_FILE = "fraud_best_ai_model.pkl"
if not os.path.exists(MODEL_FILE):
    print(f"Lỗi: Không tìm thấy file '{MODEL_FILE}'. Hãy chạy Bước 4 trước!")
    exit()

print(f"Đang nạp mô hình AI từ file: {MODEL_FILE}")
model = joblib.load(MODEL_FILE)

# 2. Kết nối SQL Server và nạp tập dữ liệu TEST thô 
# 2.1. Kết nối SQL Server
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

# 2.2. Nạp tập dữ liệu TEST
# 2.2.1 Đồng bộ hóa danh sách mục chuẩn từ tập TRAIN
categories_list = pd.read_sql("SELECT DISTINCT category FROM fraud_train",
                              con=engine)['category'].tolist()

# 2.2.2 Gọi dữ liệu từ bảng fraud_test (Tập TEST thô)
print("Đang đọc dữ liệu từ bảng 'fraud_test'")
query_test = "SELECT amt, trans_date_trans_time, category, dob, is_fraud FROM fraud_test"

df_test_raw = pd.read_sql(query_test, con=engine, chunksize=100000)
chunk = next(df_test_raw)

# 3. Các đặc trưng trên tập TEST (giống tập TRAIN ở B3)
print("Tiền xử lý và đồng bộ hóa đặc trưng cho tập TEST")
chunk['trans_date_trans_time'] = pd.to_datetime(chunk['trans_date_trans_time'])
chunk['dob'] = pd.to_datetime(chunk['dob'])

# 3.1 Khởi tạo các cột tính năng mới 
chunk['age'] = chunk['trans_date_trans_time'].dt.year - chunk['dob'].dt.year
chunk['hour'] = chunk['trans_date_trans_time'].dt.hour
chunk['is_night'] = chunk['hour'].apply(lambda x: 1 if x >= 23 or x <= 3 else 0)
chunk['is_weekend'] = chunk['trans_date_trans_time'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)

# 3.2 Tính toán tỷ lệ chi tiêu bất thường
category_avg = chunk.groupby('category')['amt'].transform('mean')
chunk['atm_to_cat_avg_ratio'] = chunk['amt'] / category_avg

# 3.3 Ép danh mục và mã hóa One-Hot Encoding 
chunk_encoded = pd.get_dummies(chunk, columns=['category'], drop_first=False)
category_cols = [col for col in chunk_encoded.columns if 'category_' in col]
chunk_encoded[category_cols] = chunk_encoded[category_cols].astype(int)

# 3.4 Loại bỏ các cột thô và chuyển kiểu dữ liệu Boolean
columns_to_drop = ['trans_date_trans_time', 'dob', 'hour']
chunk_final = chunk_encoded.drop(columns=columns_to_drop)

bool_cols = chunk_final.select_dtypes(include=['bool']).columns
chunk_final[bool_cols] = chunk_final[bool_cols].astype(int)

# 4. Chia tách và cho mô hình tiến hành thực nghiệm 
X_test_raw = chunk_final.drop(columns=['is_fraud'])
y_test = chunk_final['is_fraud']

expected_features = model.feature_names_in_
X_test = X_test_raw.reindex(columns=expected_features, fill_value=0)

print("\nMô hình đang tiến hành phân tích và chấm điểm các giao dịch mới")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# 5. Báo cáo kết quả cuối cùng 
print("Báo cáo hiệu năng kiểm thử cuối cùng (Final Evaluation)")
print(f"Accuracy (Độ chính xác tổng thể): {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (Tỷ lệ cảnh báo chính xác): {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall (Tỷ lệ tóm gọn tội phạm): {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-Score (Điểm số cân bằng hệ thống): {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC (Độ thông minh toàn diện): {roc_auc_score(y_test, y_prob):.4f}")

print("Ma trận nhầm lẫn (Confusion Matrix)")
cm = confusion_matrix(y_test, y_pred)
print(f"Số ca giao dịch AN TOÀN đoán ĐÚNG (TN): {cm[0][0]:,}")
print(f"Số ca giao dịch AN TOÀN đoán NHẦM thành lừa đảo (FP): {cm[0][1]:,}")
print(f"Số ca giao dịch LỪA ĐẢO bị BỎ SÓT (FN): {cm[1][0]:,}")
print(f"Số ca giao dịch LỪA ĐẢO tóm GỌN THÀNH CÔNG (TP): {cm[1][1]:,}")

print(f"Tổng thời gian thực hiện kiểm tra: {time.time() - start_time:.2f}")
