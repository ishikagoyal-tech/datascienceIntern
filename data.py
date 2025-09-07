# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Step 2: Load Dataset
data = pd.read_csv("transactions.csv")
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

print("Data Shape:", data.shape)
print(data.head())

# Step 3: Feature Engineering
# Latest purchase date in dataset
max_date = data['InvoiceDate'].max()

# Aggregate by customer
customer_df = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (max_date - x.max()).days,   # Recency (days since last purchase)
    'CustomerID': 'count',                               # Frequency (# transactions)
    'Amount': ['mean', 'sum']                            # Avg Order Value & Total Spend
}).reset_index()

customer_df.columns = ['CustomerID', 'Recency', 'Frequency', 'AOV', 'TotalSpend']

# Features & Target
X = customer_df[['Recency', 'Frequency', 'AOV']]
y = customer_df['TotalSpend']

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Models
# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# XGBoost
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

# Step 6: Evaluation
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

evaluate_model(y_test, rf_preds, "Random Forest")
evaluate_model(y_test, xgb_preds, "XGBoost")

# Step 7: Predict CLV for all customers
customer_df['Predicted_CLV'] = xgb.predict(X)

# Step 8: Segment Customers into Low, Medium, High
quantiles = customer_df['Predicted_CLV'].quantile([0.33, 0.66]).values

def segment_clv(value):
    if value <= quantiles[0]:
        return "Low"
    elif value <= quantiles[1]:
        return "Medium"
    else:
        return "High"

customer_df['CLV_Segment'] = customer_df['Predicted_CLV'].apply(segment_clv)

# Step 9: Save Final Results
customer_df.to_csv("Predicted_CLV.csv", index=False)
print("Final predictions saved to Predicted_CLV.csv")

# Step 10: (Optional) Visualization
plt.figure(figsize=(6,4))
sns.histplot(customer_df['Predicted_CLV'], bins=20, kde=True)
plt.title("Distribution of Predicted Customer Lifetime Value")
plt.xlabel("Predicted CLV")
plt.ylabel("Count")
plt.show()
