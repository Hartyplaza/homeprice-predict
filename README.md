# HomePrice_Predict

House price prediction using the Ames Housing dataset (79 features).

## Models
- Ridge Regression
- Lasso Regression
- XGBoost Regressor

## Metrics
RMSE, MAE, R2, SHAP for regression explainability

## Stack
pandas · scikit-learn · XGBoost · SHAP · Plotly · FastAPI · Streamlit

## Structure
data/         raw and processed datasets
notebooks/    01_eda > 02_feature_engineering > 03_modeling > 04_analysis
models/       saved model artefacts
app/          FastAPI backend + Streamlit frontend
src/          config and shared utilities
