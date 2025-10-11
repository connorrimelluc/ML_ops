import json
import numpy as np
import tensorflow as tf

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib as privacy_lib

from src.utils import load_csv, prepare_target, split, get_feature_lists
from src.cleaning import clean_v2_full

def _prep_xy(csv_path: str):
    df = load_csv(csv_path)
    df = prepare_target(df, target_col="deadlift")
    # v2 cleaning for DP run (assignment requirement)
    df = clean_v2_full(df)

    X_train, X_test, y_train, y_test = split(df, target="deadlift")

    # fit preprocessing on TRAIN only
    train_df = X_train.copy()
    train_df["deadlift"] = y_train
    num_feats, cat_feats = get_feature_lists(train_df, target="deadlift")

    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        [("num", numeric_transformer, num_feats),
         ("cat", categorical_transformer, cat_feats)],
        remainder="drop", verbose_feature_names_out=False,
    )

    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr  = preprocessor.transform(X_test)

    # ensure dense arrays for TF
    if hasattr(X_train_arr, "toarray"):
        X_train_arr = X_train_arr.toarray()
    if hasattr(X_test_arr, "toarray"):
        X_test_arr = X_test_arr.toarray()

    y_train_arr = np.asarray(y_train).astype("float32")
    y_test_arr = np.asarray(y_test).astype("float32")

    return (X_train_arr, X_test_arr, y_train_arr, y_test_arr), preprocessor

def train_eval_dp(csv_path: str,
                  report_out: str = "metrics_v2_dp.json",
                  noise_multiplier: float = 1.1,
                  l2_norm_clip: float = 1.0,
                  microbatches: int = 1,
                  epochs: int = 10,
                  batch_size: int = 64,
                  delta: float = 1e-5,
                  learning_rate: float = 0.05):
    (X_train, X_test, y_train, y_test), _ = _prep_xy(csv_path)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=microbatches,
        learning_rate=learning_rate
    )
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    rmse = float(np.sqrt(loss))

    # epsilon calc
    n = X_train.shape[0]
    eps, _ = privacy_lib.compute_dp_sgd_privacy(
        n, batch_size, noise_multiplier, epochs, delta
    )

    out = {
        "mae": float(mae),
        "rmse": rmse,
        "epsilon": float(eps),
        "delta": float(delta),
        "noise_multiplier": float(noise_multiplier),
        "l2_norm_clip": float(l2_norm_clip),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "n_train": int(n)
    }
    with open(report_out, "w") as f:
        json.dump(out, f, indent=2)
    return out
