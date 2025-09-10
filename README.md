# From Insights to Predictions: Regression Modelling of Ames Housing Prices

A comprehensive machine learning analysis predicting residential property sale prices and identifying the most important predictive features of sold house prices, using the Ames Housing dataset through exploratory data analysis, feature engineering, and iterative regression modelling.

## Project Overview

This project develops predictive models for house prices in Ames, Iowa, progressing from exploratory data analysis through sophisticated regularized regression techniques. The analysis emphasizes both predictive accuracy and model interpretability to identify key drivers of property values.

### Key Objectives
- Build predictive models using features identified through comprehensive EDA
- Compare multiple regression approaches to determine optimal modelling strategy
- Address multicollinearity and outlier influence through iterative refinement
- Identify the most important predictive features of sold house prices

## Dataset

**Source**: Ames Housing Dataset (Kaggle)
- **Size**: 2,930 residential property sales
- **Features**: 82 variables including structural, location, and sale details
- **Target**: SalePrice (residential property sale prices)
- **Location**: Ames, Iowa

## Methodology

### Data Preprocessing
- **Missing Value Treatment**: Contextual imputation (zeros for absent features, median/mode for genuine missing data)
- **Outlier Handling**: Capping extreme values at 95th-99th percentiles for select features
- **Feature Engineering**: Created composite variables including `qual_living_area_interaction` (quality × living area)

### Iterative Modeling Approach

#### Model 1: Simple Linear Regression
Individual regression models for top 5 predictors
- **Best Performer**: `qual_living_area_interaction` (Test R² = 0.796)

#### Model 2: Multiple Linear Regression  
~100 features including polynomial terms and interactions
- **Performance**: Train R² = 0.910, Test R² = 0.897
- **Issues**: Multicollinearity (VIF > 800), mild overfitting

#### Iterative Refinements:
1. **Outlier Treatment**: Removed 4 extreme observations → Test R² = 0.925
2. **Yeo-Johnson Transformation**: Improved residual normality
3. **Stepwise Feature Selection**: BIC criterion for parsimony
4. **Ridge Regularization**: Final model specification

## Results

### Final Model Performance (Ridge Regression)
- **Test R²**: 0.913
- **Test RMSE**: 0.210  
- **Test MAE**: 0.151
- **Multicollinearity**: All VIF < 10

### Top Predictive Features
1. **`qual_living_area_interaction`** (0.293) - Dominant predictor combining quality and size
2. **`house_age`** (-0.084) - Newer homes command higher prices
3. **`BsmtFin_SF_1`** (0.074) - Finished basement area
4. **`Lot_Area_capped`** (0.071) - Property lot size
5. **`remodel_age`** (-0.070) - Recent renovations increase value

### Key Insights
- Size + quality interaction consistently outperformed individual components
- Regularization emphasized generalizable structural features over location-specific effects
- Both house age and time since remodeling significantly impact valuation

## Technical Implementation

### Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy statsmodels
```

### Model Pipeline
1. Data cleaning and feature engineering
2. Polynomial transformation and scaling
3. **Cross-validation and train/test splits** (performed after feature engineering - causing data leakage)
4. **Model 1 (SLR)**: Individual simple linear regression models for top 5 predictors
5. **Model 2 (MLR)**: Multiple linear regression with ~100 features including polynomial terms
6. **Outlier removal** based on Cook's distance analysis
7. **Yeo-Johnson transformation** on target variable
8. **Stepwise BIC feature selection** for model parsimony
9. **Ridge regression** with cross-validated hyperparameter tuning

## Current Status: Exploratory Analysis

This project demonstrates comprehensive modeling techniques and achieves R² of 0.913, but requires methodological improvements before production deployment.

### Known Issues
- **Cross-validation timing**: Train-test splits performed after feature engineering, potentially overestimating performance
- **Model complexity**: Extensive polynomial features may lead to overfitting

### Next Steps for Production Use
- Implement proper data splitting workflow (split before feature engineering)
- Redesign pipeline to avoid information leakage across splits
- Validate performance using truly held-out test data
- Conduct temporal validation for future price predictions

## Repository Structure

```
├── Regression Modelling of Ames Housing Prices.ipynb
├── From Insights to Predictions. Regression Modelling of Ames Housing Prices.pdf
└── README.md
```

## Getting Started

1. **Clone Repository**
   ```bash
   git clone https://github.com/rebeccastalleymoores/ames-housing-eda-regression
   cd ames-housing-eda-regression
   ```

2. **Run Analysis**
   - Open `Regression Modelling of Ames Housing Prices.ipynb` in Jupyter
   - Execute cells sequentially for complete analysis reproduction

## Author

**Rebecca Stalley-Moores**

## License

This project is available for educational and research purposes. Please cite this work if used in academic or professional contexts.

---

*This analysis demonstrates the evolution from exploratory insights to robust predictive models, emphasizing the importance of iterative refinement in machine learning workflows.*
