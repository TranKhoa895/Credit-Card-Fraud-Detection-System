import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
'''
B4: TRAINNING MODEL
'''
start_time = time.time()

# Nạp dữ liệu từ B3
INPUT_FILE = "fraud_processed_full.csv"
if not os.path.exists(INPUT_FILE):
    print(f"Lỗi: Không tìm thấy file {INPUT_FILE}. Hãy chạy lại Bước 3 trước!")
    exit()

print(f"Đang nạp dữ liệu sạch từ file: {INPUT_FILE}...")

df = pd.read_csv(INPUT_FILE, nrows=300000)
print(df)

# Phân tách biến đầu vào x và đáp án đầu ra y
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# Chia dữ liệu thành 80% (Train) và 20% (Validation) để thử nghiệm 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
print(f"Số lượng mẫu dùng để học (Train): {X_train.shape[0]:,} dòng")
print(f"Số lượng mẫu dùng để thử nghiệm (Validation): {X_val.shape[0]:,} dòng")

# Tạo kho lưu trữ kết quả của mô hình 
results = {}

# MODEL 1: RANDOM FOREST (BASELINE MODEL)
print("\n Đang huấn luyện mô hình Random Forest")
rf_start = time.time()

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_time = time.time() - rf_start

# Dự đoán thử nghiệm 
y_pred_rf = rf_model.predict(X_val)
y_prob_rf = rf_model.predict_proba(X_val)[:,1]
results['Random Forest'] = {
    'Accuracy': accuracy_score(y_val, y_pred_rf),
    'Precision': precision_score(y_val, y_pred_rf, zero_division=0),
    'Recall': recall_score(y_val, y_pred_rf, zero_division=0),
    'F1-Score': f1_score(y_val, y_pred_rf, zero_division=0),
    'ROC-AUC': roc_auc_score(y_val, y_prob_rf),
}
print(f'Hoàn thành xong Random Forest trong: {rf_time:.2f} giây')

# MODEL 2: LIGHTGBM (BOOSTING MODEL)
print("\n Đang huấn luyện mô hình LightGBM")
lgb_start = time.time()

lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
lgb_model.fit(X_train, y_train)

lgb_time = time.time() - lgb_start

# Dự đoán thử nghiệm 
y_pred_lgb = lgb_model.predict(X_val)
y_prob_lgb = lgb_model.predict_proba(X_val)[:,1]
results['LightGBM'] = {
    'Accuracy': accuracy_score(y_val, y_pred_lgb),
    'Precision': precision_score(y_val, y_pred_lgb, zero_division=0),
    'Recall': recall_score(y_val, y_pred_lgb, zero_division=0),
    'F1-Score': f1_score(y_val, y_pred_lgb, zero_division=0),
    'ROC-AUC': roc_auc_score(y_val, y_prob_lgb),
}
print(f'Hoàn thành xong LightGBM trong: {lgb_time:.2f} giây')

# SO SÁNH HIỆU NĂNG BẮT GIAN LẬN THẺ TÍN DỤNG GIỮA HAI MÔ HÌNH
print("\n Bảng so sánh giữa hai mô hình")

df_results = pd.DataFrame(results).T
df_results['Combo_Score'] = (df_results['F1-Score'] + df_results['ROC-AUC']) / 2
pd.set_option('display.float_format', lambda x: '%.4f' % x)
print(df_results)

# Trực quan hóa mức độ quan trọng của các đặc trưng 
importances = rf_model.feature_importances_
feature_names = X.columns

df_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x='Importance', y='Feature', data=df_importance.head(10), palette='mako')
plt.title('TOP 10 FEATURE IMPORTANCES IN FRAUD DETECTION', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()

plt.savefig('fraud_feature_importance.png', dpi=300)

# Lựa chọn mô hình dựa trên cột điểm 'Combo Score' (Trung bình cộng giữa F1-Score và ROC-AUC)
best_model_name = df_results['Combo_Score'].idxmax()
print("Tiêu chí chọn: Mô hình tối ưu cả F1-Score và ROC-AUC")
print(f"Điểm số cao nhất thuộc về mô hình: {best_model_name} (Combo: {df_results.loc[best_model_name, 'Combo_Score']:.4f})")

# Đóng gói mô hình vào ổ cứng vật lý
BEST_MODEL_FILE = "fraud_best_ai_model.pkl"
if best_model_name == 'Random Forest':
    joblib.dump(rf_model, BEST_MODEL_FILE)
else:
    joblib.dump(lgb_model, BEST_MODEL_FILE)

print(f"ĐÃ LƯU BỘ THUẬT TOÁN AI TOÀN DIỆN NHẤT TẠI: {BEST_MODEL_FILE}")
print(f"Tổng thời gian chạy toàn bộ tiến trình: {time.time() - start_time:.2f} giây.")