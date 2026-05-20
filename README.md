# HomePrice_Predict

A Kaggle competition project — **House Prices: Advanced Regression Techniques**.

The goal is to predict the final sale price of residential homes in Ames, Iowa using 79 explanatory variables describing almost every aspect of each property. Submissions are evaluated on the Root Mean Squared Logarithmic Error (RMSLE) between predicted and actual sale prices.

**Live App:** [homeprice-predict-a8kcv7qpmyah8xhdi3repg.streamlit.app](https://homeprice-predict-a8kcv7qpmyah8xhdi3repg.streamlit.app)  
**Kaggle Competition:** [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

---

## Kaggle Score

| Metric | Score |
|--------|-------|
| RMSLE (public leaderboard) | **0.12427** |

---

## Solution Overview

This solution combines three models in a weighted ensemble, with extensive feature engineering and SHAP explainability for every prediction.

**Models:**
- Ridge Regression (alpha tuned via cross-validation)
- Lasso Regression (alpha tuned via cross-validation)
- XGBoost Regressor (early stopping, 3000 estimators)

**Ensemble:** Weighted blend — weights assigned inversely proportional to each model's cross-validation RMSE.

---

## Feature Engineering

- Log1p transform on target (`SalePrice`) to correct right skew
- 14 new features: `TotalSF`, `TotalBathrooms`, `HouseAge`, `QualxArea`, `QualxTotalSF`, boolean flags (`HasPool`, `HasGarage`, `HasBsmt`, `HasFireplace`, `Has2ndFloor`, `IsRemodeled`, `IsNew`), and more
- Ordinal encoding for 12 quality columns using `Ex=5` down to `Po=1`
- Log1p skewness correction on all numeric features with `|skew| > 0.75`
- Neighborhood median imputation for `LotFrontage`
- One-hot encoding for all remaining categorical features
- 2 outlier rows dropped (large `GrLivArea`, anomalously low price)

---

## Project Structure

```
HomePrice_Predict/
├── app/
│   ├── streamlit_app.py        # Streamlit frontend
│   ├── main.py                 # FastAPI backend
│   └── utils.py
├── data/
│   ├── raw/                    # train.csv, test.csv (not tracked)
│   └── processed/              # engineered features, submission
├── models/                     # saved model artifacts (.pkl)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_analysis.ipynb
├── src/
│   └── config.py               # paths, feature lists, quality mappings
├── .streamlit/
│   └── config.toml             # amber/gold dark theme
├── requirements.txt
└── runtime.txt
```

---

## App Features

The Streamlit app has four tabs:

- **Predict** — input house parameters via sliders and dropdowns, get an ensemble price prediction with a confidence range and per-model breakdown
- **Model Performance** — RMSE, MAE, R² metrics, actual vs predicted scatter, residual plot
- **SHAP Analysis** — global feature importance, dependence plots, single-prediction waterfall explanations
- **About** — competition context, Kaggle score, links

---

## Stack

| Component | Tools |
|-----------|-------|
| Models | scikit-learn, XGBoost 2.1.1 |
| Explainability | SHAP 0.49.1 |
| Visualisation | Plotly |
| Frontend | Streamlit 1.45.0 |
| Backend | FastAPI, Uvicorn |
| Language | Python 3.11 |
| Dataset | Ames Housing — Kaggle |

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/streamlit_app.py

# Run FastAPI backend (separate terminal)
uvicorn app.main:app --reload --port 8000
```

FastAPI docs available at `http://localhost:8000/docs`

---

## Author

**Ofigwe Hart** — Data Scientist / ML Engineer  
PSP Analytics Ltd | MSc Financial Engineering, WorldQuant University  

- GitHub: [github.com/Hartyplaza](https://github.com/Hartyplaza)
- LinkedIn: [linkedin.com/in/hart-ofigwe](https://linkedin.com/in/hart-ofigwe)
- Portfolio: [hartyplaza.github.io](https://hartyplaza.github.io)
