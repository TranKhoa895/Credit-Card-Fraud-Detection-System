import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns
'''
B2: EDA-Trực quan hóa các dữ liệu 
'''
# 1. Kết nối với SQL Server
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

# 2. Thiết lập giao diện biểu đồ 
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(16, 30))

# 2.1. Tỷ lệ gian lận (Q1)
query_q1 = "SELECT is_fraud, COUNT(*) AS Total FROM fraud_train GROUP BY is_fraud;"
df_q1 = pd.read_sql(query_q1, con=engine)
ax1 = plt.subplot(3, 2, 1)
ax1.pie(df_q1['Total'], labels=['Legit (0)', 'Fraud (1)'], autopct='%1.2f%%', colors=['#66b3ff','#ff9999'], startangle=90)
ax1.set_title('1. Fraud vs Legit Ratio', fontsize=12, fontweight='bold')

# 2.2. Top Categories (Q2)
query_q2 = """
SELECT category, COUNT(*) AS Total_Fraud_Cases
FROM fraud_train
WHERE is_fraud = 1
GROUP BY category
ORDER BY Total_Fraud_Cases DESC;
"""
df_q2 = pd.read_sql(query_q2, con=engine)
ax2 = plt.subplot(3, 2, 2)
sns.barplot(x='Total_Fraud_Cases', y='category', data=df_q2, palette='Reds_r', ax=ax2)
ax2.set_title('2. Top Fraud Categories', fontsize=12, fontweight='bold')

# 2.3. Giờ cao điểm (Q3)
query_q3 = """
SELECT DATEPART(HOUR, trans_date_trans_time) AS Hour, COUNT(*) AS Total_Cases 
FROM fraud_train WHERE is_fraud = 1 
GROUP BY DATEPART(HOUR, trans_date_trans_time) ORDER BY Hour;
"""
df_q3 = pd.read_sql(query_q3, con=engine)
ax3 = plt.subplot(3, 2, 3)
sns.lineplot(x='Hour', y='Total_Cases', data=df_q3, marker='o', color='crimson', ax=ax3)
ax3.set_title('3. Peak Fraud Hours', fontsize=12, fontweight='bold')
ax3.set_xticks(range(0, 24))

# 2.4. Nhóm tuổi nạn nhân (Q6)
query_q4 = """
SELECT CASE 
    WHEN YEAR(trans_date_trans_time) - YEAR(dob) < 30 THEN 'Under 30'
    WHEN YEAR(trans_date_trans_time) - YEAR(dob) BETWEEN 30 AND 50 THEN '30 - 50'
    ELSE 'Over 50' END AS Age_Group, COUNT(*) AS Total_Cases
FROM fraud_train WHERE is_fraud = 1
GROUP BY CASE 
    WHEN YEAR(trans_date_trans_time) - YEAR(dob) < 30 THEN 'Under 30'
    WHEN YEAR(trans_date_trans_time) - YEAR(dob) BETWEEN 30 AND 50 THEN '30 - 50'
    ELSE 'Over 50' END;
"""
df_q4 = pd.read_sql(query_q4, con=engine)
ax4 = plt.subplot(3, 2, 4)
sns.barplot(x='Age_Group', y='Total_Cases', data=df_q4, palette='viridis', ax=ax4)
ax4.set_title('4. Fraud cases by Age Group', fontsize=12, fontweight='bold')

# 2.5. Cuối tuần và ngày thường (Q7)
query_q5 = """
SELECT CASE WHEN DATEPART(WEEKDAY, trans_date_trans_time) IN (1, 7) THEN 'Weekend' ELSE 'Weekday' END AS Day_Type,
COUNT(*) AS Total_Cases FROM fraud_train WHERE is_fraud = 1
GROUP BY CASE WHEN DATEPART(WEEKDAY, trans_date_trans_time) IN (1, 7) THEN 'Weekend' ELSE 'Weekday' END;
"""
df_q5 = pd.read_sql(query_q5, con=engine)
ax5 = plt.subplot(3, 2, (5, 6)) # Chiếm cả hàng cuối
sns.barplot(x='Day_Type', y='Total_Cases', data=df_q5, palette='Set2', ax=ax5)
ax5.set_title('5. Fraud: Weekday vs Weekend', fontsize=12, fontweight='bold')
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.tight_layout()

plt.savefig('visualize_eda.png', dpi=300)
