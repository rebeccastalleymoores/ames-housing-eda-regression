# From Insights to Predictions: Regression Modelling of Ames Housing Prices

A comprehensive machine learning analysis predicting residential property sale prices and the most important predictive features of sold house prices, using the Ames Housing dataset through exploratory data analysis, feature engineering, and iterative regression modelling.

## 📊 Project Overview

This project develops predictive models for house prices in Ames, Iowa, progressing from exploratory data analysis through sophisticated regularized regression techniques. The analysis emphasizes both predictive accuracy and model interpretability to identify key drivers of property values.

### Key Objectives
- Build predictive models using features identified through comprehensive EDA
- Compare multiple regression approaches to determine optimal modelling strategy
- Address multicollinearity and outlier influence through iterative refinement
- Identify the most important predictive features of sold house prices
- Provide actionable insights for real estate stakeholders

## 🏠 Dataset

**Source**: Ames Housing Dataset (Kaggle)
- **Size**: 2,930 residential property sales
- **Features**: 82 variables including structural, location, and sale details
- **Target**: SalePrice (residential property sale prices)
- **Location**: Ames, Iowa

### Key Variables
- **Structural**: OverallQual, YearBuilt, GrLivArea, garage specifications
- **Location**: Neighborhood, LotConfig, zoning information  
- **Quality Indicators**: Kitchen quality, basement condition, exterior materials
- **Engineered Features**: `qual_living_area_interaction` (primary predictor)

## 🔍 Analysis Methodology

### 1. Data Preprocessing
- **Missing Value Treatment**: Contextual imputation (zeros for absent features, median/mode for genuine missing data)
- **Outlier Handling**: Capping extreme values at 95th-99th percentiles for select features
- **Feature Engineering**: Created composite variables including:
  - `qual_living_area_interaction`: Overall quality × living area
  - Temporal features: `house_age`, `remodel_age`
  - Aggregate measures: `total_bathrooms`, `total_porch_area`

### 2. Exploratory Data Analysis
- **Univariate Analysis**: Distribution assessment for numerical and categorical features
- **Bivariate Analysis**: Correlation analysis and relationship visualization
- **Hypothesis Testing**: ANOVA and Kruskal-Wallis tests validating key feature relationships

### 3. Iterative Modeling Approach

#### Model 1: Simple Linear Regression
- Individual regression models for top 5 predictors
- **Best Performer**: `qual_living_area_interaction` (Test R² = 0.796)

#### Model 2: Multiple Linear Regression  
- ~100 features including polynomial terms and interactions
- **Performance**: Train R² = 0.910, Test R² = 0.897
- **Issues**: Multicollinearity (VIF > 800), mild overfitting

#### Iterative Refinements:
1. **Outlier Treatment**: Removed 4 extreme observations based on Cook's distance
   - **Improvement**: Test R² increased to 0.925
2. **Yeo-Johnson Transformation**: Addressed heteroscedasticity
   - **Result**: Improved residual normality and variance stability
3. **Stepwise Feature Selection**: BIC criterion for parsimony
   - **Outcome**: Balanced accuracy with interpretability
4. **Ridge Regularization**: Final model specification
   - **Achievement**: Maintained accuracy while reducing multicollinearity (VIF < 10)

## 🎯 Key Results

### Final Model Performance
- **Algorithm**: Ridge Regression (built on outlier-removed model + Yeo-Johnson transformation + Stepwise BIC selection)
- **Test R²**: 0.912
- **Test RMSE**: 0.210
- **Test MAE**: 0.151
- **Multicollinearity**: All VIF < 10 (vs. 800+ in initial model)

### Top Predictive Features
1. **`qual_living_area_interaction`** (0.293) - Dominant predictor combining quality and size
2. **`house_age`** (-0.084) - Newer homes command higher prices
3. **`BsmtFin_SF_1`** (0.074) - Finished basement area
4. **`Lot_Area_capped`** (0.071) - Property lot size
5. **`remodel_age`** (-0.070) - Recent renovations increase value

### Key Insights
- **Size + Quality Interaction**: The engineered feature combining living area with overall quality consistently outperformed individual components
- **Structural Features Dominate**: Ridge regularization emphasized generalizable characteristics (basement finish, lot size, bathrooms) over neighborhood-specific effects
- **Age Effects**: Both house age and time since remodeling significantly impact valuation
- **Neighborhood Premiums**: While location matters, regularization reduced over-reliance on extreme location effects

## 🛠 Technical Implementation

### Dependencies
- **Python 3.8+**
- **Core Libraries**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: scipy, statsmodels

### Model Pipeline
1. Data cleaning and feature engineering
2. Polynomial transformation and feature scaling (RobustScaler)
3. Cross-validation approach for Model 2 (~100 features)
4. Outlier removal (4 extreme observations based on Cook's distance)
5. Yeo-Johnson transformation on target variable
6. Stepwise BIC feature selection
7. Ridge regression with cross-validated hyperparameter tuning
8. Comprehensive model diagnostics

**Note**: Train/test split was applied after preprocessing steps (polynomial transformation, scaling), causing data leakage and likely overestimating model performance on test data.

## 📈 Model Diagnostics

### Strengths
- **Strong Predictive Performance**: Explains >91% of price variance
- **Robust Feature Set**: Consistent performance across validation folds
- **Multicollinearity Resolution**: Dramatic VIF reduction through regularization
- **Interpretable Coefficients**: Clear directional relationships maintained

### Limitations & Improvements Needed
- **Data Leakage**: Transformations applied before train/test split (inflated performance metrics)
- **Outlier Sensitivity**: Only 4 observations removed may be insufficient
- **Validation**: Requires nested cross-validation for unbiased assessment

## 🔮 Future Enhancements

### Immediate Improvements
- [ ] Implement proper train/test splitting before transformations
- [ ] Nested cross-validation for robust hyperparameter tuning
- [ ] Advanced feature interactions (temporal × categorical)
- [ ] Residual analysis and robust regression techniques

### Advanced Modeling
- [ ] Tree-based methods (Random Forest, XGBoost) for nonlinear patterns
- [ ] Ensemble methods combining linear and nonlinear approaches
- [ ] Bayesian regression for uncertainty quantification
- [ ] Time series analysis for market trend incorporation

## 📁 Repository Structure

```
├── Regression Modelling of Ames Housing Prices.ipynb    # Main analysis notebook
├── From Insights to Predictions. Regression Modelling of Ames Housing Prices.pdf    # Project write up
└── README.md                             # This file
```

## 🚀 Getting Started

1. **Clone Repository**
   ```bash
   git clone https://github.com/rebeccastalleymoores/ames-housing-eda-regression
   
   cd ames-housing-eda-regression
   ```

2. **Install Dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy statsmodels
   ```

3. **Run Analysis**
   - Open 'Regression Modelling of Ames Housing Prices.ipynb` in Jupyter
   - Execute cells sequentially for complete analysis reproduction

## Current Status: Exploratory Analysis
This project demonstrates data cleaning, feature engineering, and modeling techniques on the Ames Housing dataset. The analysis identifies key property characteristics that predict sale prices and predicts sale prices with R² of 0.913 (Ridge regression) but requires additional work to address methodological issues before deployment in real-world applications.

## Known Issues

- **Cross-validation timing**: Train-test splits and cross-validation were performed after feature engineering steps that used the full dataset, potentially leading to optimistic performance estimates
- **Model complexity**: Extensive polynomial feature creation may lead to overfitting despite regularization attempts

## Next Steps for Production Use

- Implement proper data splitting workflow (split data before any feature engineering)
- Redesign feature engineering pipeline to avoid information leakage across splits
- Validate model performance using truly held-out test data
- Simplify feature set to improve model interpretability and generalization
- Conduct temporal validation if deploying for future price predictions

## Learning Outcomes

- Comprehensive data cleaning and preprocessing pipeline
- Feature engineering techniques for real estate data
- Multiple regression modeling approaches (OLS, regularized, stepwise)
- Model diagnostics and assumption testing
- Cross-validation and performance evaluation methods
- Importance of proper experimental design in machine learning workflows

## 👤 Author

**Rebecca Stalley-Moores**

## 📜 License

This project is available for educational and research purposes. Please cite this work if used in academic or professional contexts.

---

*This analysis demonstrates the evolution from exploratory insights to robust predictive models, emphasizing the importance of iterative refinement in machine learning workflows.*

