import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('./data/real_estate.csv')

print("Original DataFrame Info:")
df.info()
print("\nOriginal DataFrame Head:")
print(df.head())
print("-----"*20)

# --- Data Cleaning and Preprocessing ---

# 1. Handling Missing Values:

# Examine missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())
print("-----"*20)


#  building_construction_year: Many missing values. Instead of imputing, we will fill this with median value. 
#  Rationale: Imputation is preferred, it will preserve the data integrity better than removing this column and will keep our sample size
#  for data analysis
df['building_construction_year'] = df['building_construction_year'].fillna(df['building_construction_year'].median())

# apartment_floor: Few missing values. We can't use the same technique of filling it with the median because
# it can be any level of the building. So, we can fill the data with the average value
# Rationale: There is a chance it can have bias effect to the model if we use median and will keep the sample size
df['apartment_floor'] = df['apartment_floor'].fillna(df['apartment_floor'].mean())

# apartment_rooms, apartment_bedrooms, apartment_bathrooms: No missing data in the example given. But if there were it is best to
# fill them with median
# Rationale: Since the type of the column is integer/float and these are categorical information, therefore it is better to use
# median rather than the mean for imputation
for col in ['apartment_rooms', 'apartment_bedrooms', 'apartment_bathrooms']:
    if df[col].isnull().any():
         df[col] = df[col].fillna(df[col].median())
         
# apartment_total_area, apartment_living_area:  We will remove the text part of the string and also will
# fill missing values with the median
# Rationale: We are removing the 'm²' part to convert the column into numeric so that we can process it, imputation with median
# will preserve the data integrity and will keep the sample size for the analysis
for col in ['apartment_total_area', 'apartment_living_area']:
    # First, remove " m²"
    df[col] = df[col].str.replace(' m²', '', regex=False)
    # Second, remove thousand separators and convert to float.
    # Replacing space first handles cases like '1 000', comma handles cases like '1,000',
    # and dot handles cases like '1.000'
    df[col] = df[col].str.replace(r'[\s,.]', '', regex=True) #Remove any space/comma/dot
    df[col] = df[col].astype(float) # convert to float
    df[col] = df[col].fillna(df[col].median())


# 2. Data Type Conversion:

# Convert relevant columns to appropriate types
# Rationale: We are doing conversion as some of the operations above can affect the data type and we want to be sure that the values
# are as the desired types
df['building_construction_year'] = df['building_construction_year'].astype(int)
df['building_total_floors'] = df['building_total_floors'].astype('Int64')
df['apartment_floor'] = df['apartment_floor'].astype(int)
df['apartment_rooms'] = df['apartment_rooms'].astype(int)
df['apartment_bedrooms'] = df['apartment_bedrooms'].astype(int)
df['apartment_bathrooms'] = df['apartment_bathrooms'].astype(int)


# 3. Outlier Handling:

# For this dataset, focusing on the 'price_in_USD', 'apartment_total_area' and 'apartment_living_area'.
# Rationale: Some of the prices might be outliers which can have a negative effect to the model and will make it difficult for the model to learn
# the actual pattern, therefore it is better to remove them
def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

for col in ['price_in_USD', 'apartment_total_area', 'apartment_living_area']:
    df = remove_outliers(df, col)


# 4. Text Normalization/Cleaning
# Title: Remove redundant phrases, and convert to lowercase for consistency

# Remove the area description and convert to lower
df['title'] = df['title'].str.replace(r'(\d+\s*m²|\s*in\s*[\w\s,]+)', '', regex=True).str.lower().str.strip()
# Location: We remove the first part of the location for more consistency
df['location'] = df['location'].str.replace(r'^[^\,]+,\s*', '', regex=True).str.strip()


# 5. Identifying and Removing Duplicates
# Rationale: Duplicated rows can skew analysis, so it's important to remove them
print("\nNumber of Duplicates Before Removal:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Number of Duplicates After Removal:", df.duplicated().sum())


# 6. Display Cleaned Data
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned DataFrame Info:")
df.info()
print("\nCleaned DataFrame Head:")
print(df.head())


# 7. Save Cleaned Data to CSV
# Rationale: After all processing steps, save cleaned data to a new csv for further use.
df.to_csv('./data/cleaned_real_estate_data.csv', index=False)  
print("\nCleaned data saved to 'cleaned_real_estate_data.csv'")