import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df = pd.read_csv('cleaned_real_estate_data.csv')

# --- Exploratory Data Analysis ---

# 1. Basic Data Overview

print("-----Basic Data Overview-----")
print("\nDataFrame Info:")
df.info()
print("\nSummary Statistics:")
print(df.describe())
print("\nUnique Values in 'country' column:", df['country'].unique())
print("\nUnique Values in 'location' column:", df['location'].unique())


# 2. Distribution of Numerical Features:

print("\n-----Numerical Feature Distributions-----")
numerical_cols = ['building_construction_year', 'building_total_floors', 'apartment_floor',
                  'apartment_rooms', 'apartment_bedrooms', 'apartment_bathrooms',
                  'apartment_total_area', 'apartment_living_area', 'price_in_USD']

# Using histogram
print("\nHistograms for Numerical Features")
plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# 3. Box Plots for Numerical Features:

print("\n-----Box Plots for Numerical Features-----")
plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

# 4. Price Distribution:
print("\n-----Price Distribution-----")
plt.figure(figsize=(8, 6))
sns.histplot(df['price_in_USD'], bins=50, kde=True)
plt.title('Distribution of Price in USD')
plt.xlabel('Price in USD')
plt.ylabel('Frequency')
plt.show()

# 5. Scatter Plot of Total Area vs. Price:
print("\n-----Total Area vs. Price-----")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='apartment_total_area', y='price_in_USD', data=df)
plt.title('Total Area vs. Price')
plt.xlabel('Total Area (sqm)')
plt.ylabel('Price in USD')
plt.show()

# 6. Average price for each country
print("\n-----Average Price Per Country-----")
average_price_by_country = df.groupby('country')['price_in_USD'].mean().sort_values(ascending=False)
print(average_price_by_country)
plt.figure(figsize=(10, 6))
average_price_by_country.plot(kind='bar', color='skyblue')
plt.title('Average Price by Country')
plt.xlabel('Country')
plt.ylabel('Average Price')
plt.xticks(rotation=45, ha='right')
plt.show()

# 7. Correlation matrix heatmap of numerical data:
print("\n-----Correlation Matrix Heatmap-----")
corr = df[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# 8. Relationship between number of rooms and prices
print("\n-----Relationship between number of rooms and prices-----")
plt.figure(figsize=(10, 6))
sns.boxplot(x='apartment_rooms', y='price_in_USD', data=df)
plt.title('Price vs. Number of Rooms')
plt.xlabel('Number of Rooms')
plt.ylabel('Price in USD')
plt.show()

#9. Statistical Analysis
print("\n-----Statistical Analysis-----")
# Calculate median price by country and number of bedrooms
median_price_by_country = df.groupby('country')['price_in_USD'].median()
print("\nMedian Price Per Country:\n", median_price_by_country)
median_price_by_bedrooms = df.groupby('apartment_bedrooms')['price_in_USD'].median()
print("\nMedian Price Per Bedrooms:\n", median_price_by_bedrooms)

# Calculate mean area
mean_area = df[['apartment_total_area', 'apartment_living_area']].mean()
print("\nMean Areas: \n", mean_area)

# Relationship between total area, living area and price
print(df[['apartment_total_area', 'apartment_living_area', 'price_in_USD']].corr())