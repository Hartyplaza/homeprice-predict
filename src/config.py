import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW   = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
TRAIN_PATH = os.path.join(DATA_RAW, "train.csv")
TEST_PATH  = os.path.join(DATA_RAW, "test.csv")

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET        = "SalePrice"
LOG_TRANSFORM = True   # apply np.log1p to SalePrice before training

# ── Train / Test split ─────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── Numeric features (24) ──────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
    "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
]

# ── Categorical features (22) ──────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "MSZoning", "Street", "LotShape", "LandContour", "Utilities",
    "LotConfig", "LandSlope", "Neighborhood", "Condition1", "BldgType",
    "HouseStyle", "RoofStyle", "Exterior1st", "MasVnrType", "Foundation",
    "Heating", "CentralAir", "Electrical", "GarageType", "GarageFinish",
    "PavedDrive", "SaleType",
]

# ── Ordinal quality mapping (Ex=5 down to Po=1) ────────────────────────────────
QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}

ORDINAL_FEATURES = {
    "ExterQual" : QUALITY_MAP,
    "ExterCond" : QUALITY_MAP,
    "BsmtQual"  : QUALITY_MAP,
    "BsmtCond"  : QUALITY_MAP,
    "HeatingQC" : QUALITY_MAP,
    "KitchenQual": QUALITY_MAP,
    "FireplaceQu": QUALITY_MAP,
    "GarageQual" : QUALITY_MAP,
    "GarageCond" : QUALITY_MAP,
    "PoolQC"     : QUALITY_MAP,
    "OverallQual" : {i: i for i in range(1, 11)},
    "OverallCond" : {i: i for i in range(1, 11)},
}
