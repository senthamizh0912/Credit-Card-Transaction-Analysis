import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay

import kagglehub



path = kagglehub.dataset_download("ybifoundation/credit-card-transaction")

print("Path to dataset files:", path)

df = pd.read_csv(f"{path}/CreditCardTransaction.csv")
df.head()


df['TranxDate'] = pd.to_datetime(df['TranxDate'], errors='coerce')

print(df.isnull().sum())


df = df.dropna(subset=['TranxDate'])

print(df.describe())
print(df['Department'].value_counts())

dept_spend = df.groupby('Department')['TrnxAmount'].sum().sort_values(ascending=False)
dept_spend.plot(kind='bar', figsize=(10,6), color='teal')
plt.title("Total Spending by Department")
plt.ylabel("Amount ($)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

monthly_spend = df.groupby(['Year', 'Month'])['TrnxAmount'].sum()

monthly_spend.plot(kind='line', marker='o', figsize=(10,5))
plt.title("Monthly Spending Over Time")
plt.ylabel("Amount ($)")
plt.grid(True)
plt.show()

top_merchants = df['Merchant'].value_counts().head(10)

top_merchants.plot(kind='bar', color='orange')
plt.title("Top 10 Frequent Merchants")
plt.ylabel("Transaction Count")
plt.xticks(rotation=45)
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

df_cluster = df[['TrnxAmount', 'Department']].copy()
le = LabelEncoder()
df_cluster['Department'] = le.fit_transform(df_cluster['Department'])


kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['Cluster'] = kmeans.fit_predict(df_cluster)


sns.scatterplot(x='TrnxAmount', y='Department', hue='Cluster', data=df_cluster, palette='viridis')
plt.title("KMeans Clustering of Transactions")
plt.show()

