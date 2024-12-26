# Real Estate Data Analysis Project

## Overview
This project analyzes international real estate data to uncover market trends, property characteristics, and price patterns across different countries. It implements a complete data science pipeline including data processing, exploratory data analysis, and machine learning models for price prediction.

## Project Structure
```
├── crop_huge_data.py          # Script to create manageable dataset
├── data_processing.py         # Data cleaning and preprocessing
├── data_analysis.py          # Exploratory data analysis
├── machine_learning.py       # ML models implementation
├── requirements.txt          # Project dependencies
└── data/
    ├── world_real_estate_data(147k).csv    # Original dataset
    ├── real_estate.csv                     # Cropped dataset
    └── cleaned_real_estate_data.csv        # Processed dataset
```

## Features
1. **Data Processing**
   - Handles missing values using appropriate imputation strategies
   - Removes outliers using IQR method
   - Cleans and standardizes text data
   - Converts measurements to consistent units
   - Handles duplicate entries

2. **Exploratory Data Analysis**
   - Statistical analysis of property features
   - Price distribution analysis
   - Correlation analysis between variables
   - Geographic price variation analysis
   - Property characteristics visualization

3. **Machine Learning Models**
   - Linear Regression for baseline predictions
   - Random Forest Regressor for advanced modeling
   - Feature importance analysis
   - Model performance comparison
   - Prediction visualization

## Key Findings
- Price variations across different countries and regions
- Correlation between property features and prices
- Impact of property characteristics on value
- Model performance metrics for price prediction

## Technical Details
### Data Processing
- Missing value imputation using median/mean strategies
- Outlier removal using IQR method
- Text standardization for categorical variables
- Unit conversion for consistent measurements

### Machine Learning
- Train-test split: 80-20
- Feature engineering for categorical variables
- Model evaluation using RMSE and R² metrics
- Cross-validation for robust performance assessment

## Installation & Usage
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the scripts in order:
   ```bash
   python crop_huge_data.py
   python data_processing.py
   python data_analysis.py
   python machine_learning.py
   ```

## Requirements
- Python 3.8+
- See requirements.txt for package dependencies