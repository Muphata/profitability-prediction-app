# ============================================
# Train Final Model - Profitability Prediction App
# ============================================

# Step 1: Import Libraries
# --------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 2: Load the Dataset
# --------------------------------------------
data = pd.read_csv('dataset.csv', encoding='latin1')

# Step 3: Feature Engineering (Create Days_to_Ship)
# --------------------------------------------
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Ship Date'] = pd.to_datetime(data['Ship Date'])
data['Days_to_Ship'] = (data['Ship Date'] - data['Order Date']).dt.days

# Step 4: Select Features and Target
# --------------------------------------------
features = ['Sales', 'Discount', 'Quantity', 
            'Category', 'Sub-Category', 'Ship Mode', 
            'Region', 'State', 'City', 'Days_to_Ship']
target = 'Profit'

# Check if all required columns exist
if not all(col in data.columns for col in features + [target]):
    raise ValueError("❌ Error: Some columns are missing. Check your CSV file.")

X = data[features]
y = data[target]

# Step 5: One-Hot Encode Categorical Features
# --------------------------------------------
categorical_features = ['Category', 'Sub-Category', 'Ship Mode', 'Region', 'State', 'City']

X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Step 6: Split the Data
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Step 7: Train the Random Forest Model
# --------------------------------------------
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate the Model
# --------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n🔍 Model Evaluation:")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"MSE (Mean Squared Error): {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 9: Save the Trained Model
# --------------------------------------------
joblib.dump(model, 'model.joblib')

print("\n✅ Final model trained and saved successfully as 'model.joblib'!")
