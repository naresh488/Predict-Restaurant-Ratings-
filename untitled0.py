# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19Js6BrjFPw0PeOgQ8o-qeR24-MjAdCnb
"""

import pandas as pd
import io
from google.colab import files

uploaded = files.upload()

# Retrieve the filename of the first uploaded file
file_name = next(iter(uploaded))

# Load the dataset into a DataFrame
data = pd.read_csv(io.BytesIO(uploaded[file_name]))

# Display the first few rows to make sure it's loaded correctly
data.head()

# Display the first few rows of the DataFrame
print(data.head())

# Display the summary of the DataFrame
print(data.info())

# Generate descriptive statistics for numeric columns
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Checking if 'Aggregate_rating' is in the list of numerical columns and remove it if present
if 'Aggregate_rating' in numerical_cols:
    numerical_cols.remove('Aggregate_rating')

# Recreate the lists of categorical and numerical columns directly from the DataFrame
categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
numerical_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]

# Ensure 'Aggregate_rating' is only included in the target variable, not in the features
numerical_cols = [col for col in numerical_cols if col != 'Aggregate_rating']

# Print all column names in the DataFrame
print(data.columns)

# Strip extra spaces from column names
data.columns = data.columns.str.strip()

# Print column names to confirm the exact name of all columns
print(list(data.columns))

# Define the target variable
y = data['Aggregate rating']  # Ensure the name is exactly as shown in the list

# Define the features by dropping the target column
X = data.drop('Aggregate rating', axis=1)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(data.head())
print(data.info())

# Define the target variable
y = data['Aggregate rating']  # Corrected column name

# Define the features by dropping the target column
X = data.drop('Aggregate rating', axis=1)  # Corrected column name

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Identify the categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Define a ColumnTransformer to apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_train.select_dtypes(exclude=['object']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create a pipeline that includes both the preprocessor and the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Train the model
model.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("Cross-validated R-squared:", scores.mean())

from sklearn.linear_model import Ridge, Lasso

# Example using Ridge Regression
ridge_model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', Ridge())])
ridge_model.fit(X_train, y_train)
scores = cross_val_score(ridge_model, X, y, cv=5, scoring='r2')
print("Cross-validated R-squared with Ridge:", scores.mean())

lasso_model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', Lasso())])
lasso_model.fit(X_train, y_train)
scores = cross_val_score(lasso_model, X, y, cv=5, scoring='r2')
print("Cross-validated R-squared with Lasso:", scores.mean())

from sklearn.model_selection import GridSearchCV

param_grid = {'regressor__alpha': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(ridge_model, param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)

print("Best alpha parameter:", grid.best_params_)
print("Best cross-validated R-squared:", grid.best_score_)

final_ridge_model = Ridge(alpha=10)
final_model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', final_ridge_model)])
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test Mean Squared Error:", mse)
print("Test R-squared:", r2)

print("Model Coefficients:", final_ridge_model.coef_)

import joblib
joblib.dump(final_model, 'ridge_regression_model.pkl')