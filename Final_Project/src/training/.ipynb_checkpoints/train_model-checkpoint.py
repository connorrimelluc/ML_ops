import os
import pandas as pd
import joblib

from flaml import AutoML
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- NEW: import wrapper from the proper module ---
from pathlib import Path
import sys

# Add src/ to import path
SRC_DIR = Path(__file__).resolve().parents[1]   # .../Final_Project/src
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models.wrapper import PreprocessAndAutoML   # <-- proper module import


DATA_DIR = "Final_Project/data"
MODEL_DIR = "Final_Project/models"


def load_data():
    X_train = pd.read_parquet(os.path.join(DATA_DIR, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(DATA_DIR, "y_train.parquet"))["Price"]
    return X_train, y_train


def build_preprocess_pipeline(X):
    # Adjust these if your column names differ
    numeric_cols = ["Year", "Engine Size", "Mileage"]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, y_train = load_data()

    # 1) Fit preprocessor on raw training data
    preprocessor = build_preprocess_pipeline(X_train)
    preprocessor.fit(X_train)

    # 2) Transform X_train
    X_train_proc = preprocessor.transform(X_train)

    # 3) Run FLAML AutoML on the transformed data
    automl = AutoML()
    automl_settings = {
        "time_budget": 60,       # seconds
        "metric": "rmse",
        "task": "regression",
        "log_file_name": "automl.log",
        "estimator_list": ["lgbm", "rf", "extra_tree"],  # exclude xgboost
    }

    automl.fit(X_train=X_train_proc, y_train=y_train, **automl_settings)

    # 4) Wrap everything into a single object for inference
    model = PreprocessAndAutoML(preprocessor, automl)

    out_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(model, out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()