
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---- I/O helpers ----
def load_data(train_path="../data/claims_train.csv", test_path="../data/claims_test.csv"):
    train = pd.read_csv(train_path)
    try:
        test = pd.read_csv(test_path)
    except Exception:
        test = None
    return train, test

#this one just makes sures you got the dataset

def save_model(model, path="models/lightgbm_model.joblib"):
    joblib.dump(model, path)
    return path

def load_model(path="models/lightgbm_model.joblib"):
    return joblib.load(path)

#these just save and load traned model - useful for reproducibility and for sharing results

# ---- evaluation ----
def poisson_deviance(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), eps)
    term = y_true * np.log((y_true + eps) / y_pred) - (y_true - y_pred)
    return 2.0 * np.mean(term)
#Poisson deviance is the loss appropiate for count data - this computes a mean deviance (smaller = better)


def evaluate_counts(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "poisson_deviance": float(poisson_deviance(y_true, y_pred)),
        "total_true": float(np.sum(y_true)),
        "total_pred": float(np.sum(y_pred))
    }

#this one is just for more convenience, it returns RMSE, MAE, Poisson deviance and total counts

# ---- preprocessing ----
def preprocess_train(df,
                     target_col="ClaimNb",
                     exposure_col="Exposure",
                     exposure_cap=1.0,
                     id_cols=None,
                     cat_threshold=50):
    """
    Minimal preprocessing:
      - Keep original exposure
      - Flag and cap exposure
      - Add log_exposure
      - Fill numeric NA with median, categorical NA with "MISSING"
      - Detect categorical columns
      - Fit an OrdinalEncoder and return encoder (for re-use on test)
    Returns: X (DataFrame), y (Series), encoder, cat_cols, num_cols
    """
    df = df.copy()
    if id_cols is None:
        id_cols = ["IDpol", "Id", "PolicyID"]
    df["Exposure_orig"] = df[exposure_col]
    df["exposure_large"] = (df[exposure_col] > exposure_cap).astype(int)
    df[exposure_col] = df[exposure_col].clip(upper=exposure_cap)
    df["log_exposure"] = np.log(df[exposure_col].replace(0, 1e-6))

    # fill missing
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("MISSING")
    # simple numeric fills
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())

    # prepare feature list
    exclude = set([target_col, "Exposure_orig", exposure_col] + id_cols)
    features = [c for c in df.columns if c not in exclude and c != target_col]

    # categorical detection
    cat_cols = [c for c in features if (df[c].dtype == "object" or df[c].nunique() <= cat_threshold)]
    num_cols = [c for c in features if c not in cat_cols]

    # fit ordinal encoder
    enc = None
    if len(cat_cols):
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        enc.fit(df[cat_cols].astype(str))
        df[cat_cols] = enc.transform(df[cat_cols].astype(str))

    X = df[features]
    y = df[target_col]
    return X, y, enc, cat_cols, num_cols

def preprocess_test(df, encoder, features, cat_cols, num_cols, exposure_col="Exposure", exposure_cap=1.0):
    """
    Apply same transforms to test set. Returns X_test DataFrame.
    """
    df = df.copy()
    df["Exposure_orig"] = df[exposure_col]
    df["exposure_large"] = (df[exposure_col] > exposure_cap).astype(int)
    df[exposure_col] = df[exposure_col].clip(upper=exposure_cap)
    df["log_exposure"] = np.log(df[exposure_col].replace(0, 1e-6))

    # fill
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("MISSING")

    # encode
    if encoder is not None and len(cat_cols):
        df[cat_cols] = encoder.transform(df[cat_cols].astype(str))

    X = df[features]
    return X

# ---- train/val split helper ----
def get_train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
