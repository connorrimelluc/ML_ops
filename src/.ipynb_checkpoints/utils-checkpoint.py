import pandas as pd
from sklearn.model_selection import train_test_split

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def prepare_target(df: pd.DataFrame, target_col: str = "deadlift") -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataset.")
    return df

def split(df: pd.DataFrame, target: str = "deadlift", test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_feature_lists(df: pd.DataFrame, target: str = "deadlift"):
    """Return lists of numeric and categorical feature names (excluding target)."""
    feature_df = df.drop(columns=[target])
    num_feats = feature_df.select_dtypes(include=["number"]).columns.tolist()
    cat_feats = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    return num_feats, cat_feats