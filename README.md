# Rental Price Prediction

This project aims to build a predictive model capable of **estimating the rental value of properties** based on variables such as location, property characteristics, and additional attributes provided in the dataset.

## 1. Project Objective

The central objective is to offer models that assist in property pricing, allowing:

- Identify factors that most influence rental prices;  
- Reliably predict rental values of new properties;  
- Compare different algorithms and modeling strategies;  
- Support decision-making in the real estate market.  

## 2. Data Preparation

- Creation of the transformed target variable `log_rent` (logarithm of rental value) to correct asymmetry;  
- Treatment of missing values;  
- Encoding of categorical variables and standardization of numerical variables;  
- Exploratory data analysis (EDA) to identify outliers and distribution patterns;  
- Split of data into training and test sets for model evaluation.  

## 3. Results

The following regression models were evaluated:

- Linear Regression (OLS)  
- Ridge and Lasso Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  

The metrics considered were **R²**, **MAE**, and **RMSE**, both on the logarithmic scale and on the original scale of rental values.  

| Model                  | R²     | MAE (R$) | RMSE (R$) | Notes |
|------------------------|--------|----------|-----------|-------|
| Linear Regression (rent) | 0.625  | 883.50   | 1,397.46  | Without log-transform |
| Linear Regression (log)  | 0.710  | 1,306.20 | 0.33 (log)| Better fit with log |
| Decision Tree           | 0.9934 | 67.45    | 182.87    | Strong overfitting |
| Random Forest           | 0.9966 | 35.58    | 131.32    | Best balance |
| Gradient Boosting       | 0.9916 | 116.48   | 205.67    | Good performance |

Highlight: **Random Forest** showed the best balance between performance and generalization, achieving **R² ≈ 0.997** and mean absolute error around **R$ 35**.  

## 4. Modeling Strategies

- Detailed exploratory analysis and treatment of the response variable;  
- Comparison between linear and tree-based models;  
- Evaluation with standardized metrics;  
- Interpretation of residuals and analysis of most important variables.  

## 5. Tools Used

- Python 3.12;  
- Jupyter Notebook (executed via VSCode);  
- Libraries:
  - `pandas` and `numpy` for data manipulation;  
  - `matplotlib` and `seaborn` for visualization;  
  - `scikit-learn` and `statsmodels` for modeling;  
  - `streamlit` for interactive web application;  

## 6. Interactive Streamlit Application

This project includes an interactive web application built with Streamlit that provides:

- **Home**: Project overview, objectives, and technologies used
- **Data Exploration**: Sample data visualization, descriptive statistics, distribution plots, and correlation heatmap
- **Model Performance**: Model comparison with metrics (R², MAE, RMSE), performance charts, and mathematical formulas
- **Make Predictions**: Interactive form to input property characteristics and get instant rental price predictions with confidence intervals

### Running the Application

To run the Streamlit application:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 7. Possible Improvements

- Implementation of a complete `Pipeline` with `ColumnTransformer`;  
- Application of cross-validation for greater robustness;  
- Inclusion of variable selection techniques and multicollinearity analysis;  
- Testing with additional algorithms (XGBoost, LightGBM, CatBoost);  
- Development of a deployable model (API or interactive dashboard).  

## Author

**Ricardo Luís Bertolucci Filho**  

- [LinkedIn](https://www.linkedin.com/in/ricardo-lu%C3%ADs-bertolucci-filho/)  
- [GitHub](https://github.com/ric-rky/ric-rky)  
- E-mail: bertolucci.rl@gmail.com  

##

Main Notebook: [`projeto_aluguel.ipynb`](https://github.com/ric-rky/Predicao-de-valores-de-aluguel/blob/main/projeto_aluguel.ipynb)
