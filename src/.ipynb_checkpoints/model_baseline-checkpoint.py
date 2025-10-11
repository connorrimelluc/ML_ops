import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.utils import load_csv, prepare_target, split, get_feature_lists
from src.cleaning import clean_v1_minimal  # NEW

def train_eval(csv_path: str, report_out: str = "metrics_v1.json"):
    """Baseline regression with auto preprocessing for mixed dtypes."""
    df = load_csv(csv_path)
    df = prepare_target(df, target_col="deadlift")

    # >>> minimal cleaning so y has no NaNs
    df = clean_v1_minimal(df)

    # Split first (avoid leakage)
    X_train, X_test, y_train, y_test = split(df, target="deadlift")

    # Determine feature lists from TRAIN slice
    train_df = X_train.copy()
    train_df["deadlift"] = y_train
    num_feats, cat_feats = get_feature_lists(train_df, target="deadlift")

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_feats),
            ("cat", categorical_transformer, cat_feats),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = Pipeline(steps=[("preprocess", preprocessor), ("reg", LinearRegression())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "n_features_numeric": len(num_feats),
        "n_features_categorical": len(cat_feats),
        "n_rows_train": int(len(X_train)),
        "n_rows_test": int(len(X_test)),
    }

    with open(report_out, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics