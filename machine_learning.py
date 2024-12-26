import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load the cleaned data
df = pd.read_csv('./data/cleaned_real_estate_data.csv')


# --- Feature Engineering/Selection ---
# We will keep all the columns for this example except for title and url
# 1. Handling Text columns
# Drop the columns which we will not be using for the ML part.
df_ml = df.drop(columns=['title', 'url'])

# 2. Separating numerical and categorical features for preprocessing
numerical_features = df_ml.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df_ml.select_dtypes(include=['object']).columns.tolist()

# Removing 'price_in_USD' from numerical features if it's present, since this is the target column
if 'price_in_USD' in numerical_features:
  numerical_features.remove('price_in_USD')


# 3. Preprocessing using pipeline

# Numerical Pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with median
    ('scaler', StandardScaler())
])

# Categorical Pipeline
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine them using Column Transformer
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])


# --- Train/Test Split ---
X = df_ml.drop('price_in_USD', axis=1)  # Features
y = df_ml['price_in_USD']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# --- Model Training and Evaluation ---

# 1. Linear Regression
print("\n-----Linear Regression-----")
lr_model = LinearRegression()
lr_model.fit(X_train_transformed, y_train)
lr_predictions = lr_model.predict(X_test_transformed)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_r2 = r2_score(y_test, lr_predictions)
print(f"Linear Regression RMSE: {lr_rmse:.2f}")
print(f"Linear Regression R2 Score: {lr_r2:.2f}")



# 2. Random Forest Regressor
print("\n-----Random Forest Regressor-----")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_transformed, y_train)
rf_predictions = rf_model.predict(X_test_transformed)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)
print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"Random Forest R2 Score: {rf_r2:.2f}")

# --- Model Comparison ---
print("\n-----Model Comparison-----")
print("RMSE Comparison:")
print(f"Linear Regression: {lr_rmse:.2f}")
print(f"Random Forest: {rf_rmse:.2f}")
print("\nR2 Score Comparison:")
print(f"Linear Regression: {lr_r2:.2f}")
print(f"Random Forest: {rf_r2:.2f}")


if lr_rmse < rf_rmse:
    print("\nLinear Regression has better RMSE.")
else:
    print("\nRandom Forest has better RMSE.")

if lr_r2 > rf_r2:
    print("\nLinear Regression has better R2 Score.")
else:
    print("\nRandom Forest has better R2 Score.")


# --- Visualization ---

# 1. Scatter plot of Predicted vs. Actual Prices
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_predictions)
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('Linear Regression: Actual vs. Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) #Line of perfect prediction

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_predictions)
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('Random Forest: Actual vs. Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) #Line of perfect prediction

plt.tight_layout()
plt.show()

# 2. Residual Plots:
plt.figure(figsize=(12, 6))

lr_residuals = y_test - lr_predictions

plt.subplot(1, 2, 1)
plt.scatter(lr_predictions, lr_residuals)
plt.xlabel('Predicted Price (USD)')
plt.ylabel('Residuals')
plt.title('Linear Regression Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')

rf_residuals = y_test - rf_predictions

plt.subplot(1, 2, 2)
plt.scatter(rf_predictions, rf_residuals)
plt.xlabel('Predicted Price (USD)')
plt.ylabel('Residuals')
plt.title('Random Forest Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')

plt.tight_layout()
plt.show()


# 3. Histograms of Predictions

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(lr_predictions, kde=True)
plt.title('Linear Regression Predictions Distribution')
plt.xlabel('Predicted Price')

plt.subplot(1, 2, 2)
sns.histplot(rf_predictions, kde=True)
plt.title('Random Forest Predictions Distribution')
plt.xlabel('Predicted Price')

plt.tight_layout()
plt.show()

# 4. R2 score bar chart
r2_scores = {'Linear Regression': lr_r2, 'Random Forest': rf_r2}
models = list(r2_scores.keys())
values = list(r2_scores.values())

plt.figure(figsize=(8, 6))
plt.bar(models, values, color=['skyblue', 'lightcoral'])
plt.title('R2 Score Comparison')
plt.ylabel('R2 Score')
plt.ylim(0, 1)
plt.show()