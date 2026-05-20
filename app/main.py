"""
HomePrice_Predict — FastAPI Backend
Author : Ofigwe Hart
Project: HomePrice_Predict
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HomePrice_Predict API",
    description="Ensemble regression API for Ames Housing price prediction. "
                "Combines Ridge, Lasso, and XGBoost with SHAP explainability.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load artifacts ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")

try:
    ridge_pipe  = joblib.load(os.path.join(MODEL_DIR, "ridge_model.pkl"))
    lasso_pipe  = joblib.load(os.path.join(MODEL_DIR, "lasso_model.pkl"))
    xgb_model   = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    weights     = joblib.load(os.path.join(MODEL_DIR, "ensemble_weights.pkl"))
    feat_cols   = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    X_train     = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train     = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model artifact not found: {e}. "
        "Run notebooks 01–03 first to generate all artifacts."
    )

QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}

NEIGHBORHOODS = [
    "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt",
    "Gilbert", "Sawyer", "NWAmes", "SawyerW", "BrkSide", "Crawfor",
    "Mitchel", "NoRidge", "Timber", "IDOTRR", "ClearCr", "StoneBr",
    "SWISU", "Blmngtn", "MeadowV", "BrDale", "Veenker", "NPkVill", "Blueste"
]


# ── Request / Response schemas ────────────────────────────────────────────────
class HouseFeatures(BaseModel):
    # Size
    gr_liv_area   : int   = Field(1500,  ge=300,  le=5000,  description="Above grade living area (sq ft)")
    total_bsmt_sf : int   = Field(800,   ge=0,    le=3000,  description="Total basement SF")
    first_flr_sf  : int   = Field(1000,  ge=300,  le=4000,  description="First floor SF")
    garage_area   : int   = Field(400,   ge=0,    le=1500,  description="Garage area (sq ft)")
    garage_cars   : int   = Field(2,     ge=0,    le=4,     description="Garage capacity (cars)")

    # Age
    year_built    : int   = Field(1990,  ge=1872, le=2010,  description="Year built")
    year_remod    : int   = Field(1995,  ge=1950, le=2010,  description="Year remodelled")

    # Condition
    overall_qual  : int   = Field(6,     ge=1,    le=10,    description="Overall quality (1-10)")
    overall_cond  : int   = Field(5,     ge=1,    le=10,    description="Overall condition (1-10)")

    # Rooms
    full_bath     : int   = Field(2,     ge=0,    le=4,     description="Full bathrooms")
    half_bath     : int   = Field(0,     ge=0,    le=2,     description="Half bathrooms")
    bedroom       : int   = Field(3,     ge=0,    le=6,     description="Bedrooms above grade")
    tot_rms       : int   = Field(7,     ge=2,    le=14,    description="Total rooms above grade")
    fireplaces    : int   = Field(0,     ge=0,    le=3,     description="Number of fireplaces")

    # Quality ratings
    exter_qual    : str   = Field("Gd",  description="Exterior quality: Ex, Gd, TA, Fa, Po")
    kitchen_qual  : str   = Field("Gd",  description="Kitchen quality: Ex, Gd, TA, Fa, Po")
    bsmt_qual     : str   = Field("TA",  description="Basement quality: Ex, Gd, TA, Fa, Po, None")

    # Location
    neighborhood  : str   = Field("NAmes", description="Neighborhood name")

    class Config:
        schema_extra = {
            "example": {
                "gr_liv_area"  : 1800,
                "total_bsmt_sf": 900,
                "first_flr_sf" : 1000,
                "garage_area"  : 480,
                "garage_cars"  : 2,
                "year_built"   : 2003,
                "year_remod"   : 2003,
                "overall_qual" : 7,
                "overall_cond" : 5,
                "full_bath"    : 2,
                "half_bath"    : 1,
                "bedroom"      : 3,
                "tot_rms"      : 8,
                "fireplaces"   : 1,
                "exter_qual"   : "Gd",
                "kitchen_qual" : "Gd",
                "bsmt_qual"    : "Gd",
                "neighborhood" : "CollgCr"
            }
        }


class PredictionResponse(BaseModel):
    predicted_price : float
    price_low       : float
    price_high      : float
    ridge_price     : float
    lasso_price     : float
    xgb_price       : float
    ensemble_weights: dict
    top_shap_factors: list


class ModelInfoResponse(BaseModel):
    model_version   : str
    training_samples: int
    feature_count   : int
    ensemble_weights: dict
    models          : list


class HealthResponse(BaseModel):
    status  : str
    message : str


# ── Feature builder ────────────────────────────────────────────────────────────
def build_feature_row(h: HouseFeatures) -> pd.DataFrame:
    yr_sold = 2010

    house_age       = max(yr_sold - h.year_built, 0)
    yrs_since_remod = max(yr_sold - h.year_remod, 0)
    is_remodeled    = int(h.year_built != h.year_remod)
    is_new          = int(h.year_built == yr_sold)
    total_sf        = h.total_bsmt_sf + h.first_flr_sf
    total_bathrooms = h.full_bath + 0.5 * h.half_bath
    qual_x_area     = h.overall_qual * h.gr_liv_area
    qual_x_total_sf = h.overall_qual * total_sf

    row = {
        "GrLivArea"         : np.log1p(h.gr_liv_area),
        "TotalBsmtSF"       : np.log1p(h.total_bsmt_sf),
        "1stFlrSF"          : np.log1p(h.first_flr_sf),
        "GarageArea"        : np.log1p(h.garage_area),
        "GarageCars"        : h.garage_cars,
        "FullBath"          : h.full_bath,
        "HalfBath"          : h.half_bath,
        "BedroomAbvGr"      : h.bedroom,
        "TotRmsAbvGrd"      : h.tot_rms,
        "Fireplaces"        : h.fireplaces,
        "OverallQual"       : h.overall_qual,
        "OverallCond"       : h.overall_cond,
        "ExterQual"         : QUALITY_MAP.get(h.exter_qual,   3),
        "KitchenQual"       : QUALITY_MAP.get(h.kitchen_qual, 3),
        "BsmtQual"          : QUALITY_MAP.get(h.bsmt_qual,    3),
        "TotalSF"           : np.log1p(total_sf),
        "TotalBathrooms"    : total_bathrooms,
        "HouseAge"          : np.log1p(house_age),
        "YearsSinceRemodel" : np.log1p(yrs_since_remod),
        "IsRemodeled"       : is_remodeled,
        "IsNew"             : is_new,
        "QualxArea"         : np.log1p(qual_x_area),
        "QualxTotalSF"      : np.log1p(qual_x_total_sf),
        "HasGarage"         : int(h.garage_area > 0),
        "HasBsmt"           : int(h.total_bsmt_sf > 0),
        "HasFireplace"      : int(h.fireplaces > 0),
        "YrSold"            : yr_sold,
        "YearBuilt"         : h.year_built,
        "YearRemodAdd"      : h.year_remod,
    }

    # One-hot: Neighborhood
    for nbhd in NEIGHBORHOODS:
        row[f"Neighborhood_{nbhd}"] = int(h.neighborhood == nbhd)

    df = pd.DataFrame([row])

    # Align to training columns — fill missing with 0
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feat_cols]

    return df


def ensemble_predict(row: pd.DataFrame):
    r = ridge_pipe.predict(row)[0]
    l = lasso_pipe.predict(row)[0]
    x = xgb_model.predict(row)[0]
    ens = weights[0]*r + weights[1]*l + weights[2]*x
    return r, l, x, ens


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def root():
    return {
        "status" : "ok",
        "message": "HomePrice_Predict API is running. POST to /predict to get a price estimate."
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return {"status": "ok", "message": "All artifacts loaded successfully."}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Info"])
def model_info():
    return {
        "model_version"   : "1.0.0",
        "training_samples": int(len(X_train)),
        "feature_count"   : int(len(feat_cols)),
        "ensemble_weights": {
            "Ridge"  : round(float(weights[0]), 4),
            "Lasso"  : round(float(weights[1]), 4),
            "XGBoost": round(float(weights[2]), 4),
        },
        "models": ["Ridge (RobustScaler pipeline)", "Lasso (RobustScaler pipeline)", "XGBoost Regressor"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(house: HouseFeatures):
    # Validate quality inputs
    valid_qual = {"Ex", "Gd", "TA", "Fa", "Po"}
    if house.exter_qual not in valid_qual:
        raise HTTPException(status_code=422, detail=f"exter_qual must be one of {valid_qual}")
    if house.kitchen_qual not in valid_qual:
        raise HTTPException(status_code=422, detail=f"kitchen_qual must be one of {valid_qual}")
    if house.bsmt_qual not in (valid_qual | {"None"}):
        raise HTTPException(status_code=422, detail=f"bsmt_qual must be one of {valid_qual | {'None'}}")
    if house.neighborhood not in NEIGHBORHOODS:
        raise HTTPException(status_code=422, detail=f"neighborhood must be one of {NEIGHBORHOODS}")

    try:
        row = build_feature_row(house)
        r_log, l_log, x_log, ens_log = ensemble_predict(row)

        price     = float(np.expm1(ens_log))
        r_price   = float(np.expm1(r_log))
        l_price   = float(np.expm1(l_log))
        x_price   = float(np.expm1(x_log))
        price_low  = price * 0.90
        price_high = price * 1.10

        # SHAP top factors for this prediction
        try:
            import shap
            explainer   = shap.TreeExplainer(xgb_model)
            shap_vals   = explainer.shap_values(row)
            shap_series = pd.Series(shap_vals[0], index=feat_cols)
            top_shap    = (
                shap_series.abs()
                .sort_values(ascending=False)
                .head(5)
            )
            top_factors = [
                {"feature": feat, "shap_value": round(float(shap_series[feat]), 4)}
                for feat in top_shap.index
            ]
        except Exception:
            top_factors = []

        return {
            "predicted_price" : round(price, 2),
            "price_low"       : round(price_low, 2),
            "price_high"      : round(price_high, 2),
            "ridge_price"     : round(r_price, 2),
            "lasso_price"     : round(l_price, 2),
            "xgb_price"       : round(x_price, 2),
            "ensemble_weights": {
                "Ridge"  : round(float(weights[0]), 4),
                "Lasso"  : round(float(weights[1]), 4),
                "XGBoost": round(float(weights[2]), 4),
            },
            "top_shap_factors": top_factors,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/neighborhoods", tags=["Info"])
def get_neighborhoods():
    return {"neighborhoods": NEIGHBORHOODS}


@app.get("/quality-options", tags=["Info"])
def get_quality_options():
    return {
        "options"    : ["Ex", "Gd", "TA", "Fa", "Po"],
        "descriptions": {
            "Ex": "Excellent",
            "Gd": "Good",
            "TA": "Typical / Average",
            "Fa": "Fair",
            "Po": "Poor"
        }
    }
