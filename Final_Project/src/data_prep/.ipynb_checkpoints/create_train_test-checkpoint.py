import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths are relative to the repo root (~/ML_ops)
INPUT = "Final_Project/data/car_price_prediction_.csv"
OUTPUT_DIR = "Final_Project/data"

def main():
    df = pd.read_csv(INPUT)

    # Drop identifier if present
    if "Car ID" in df.columns:
        df = df.drop(columns=["Car ID"])

    # Target variable
    y = df["Price"]
    X = df.drop(columns=["Price"])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save X as parquet DataFrames
    X_train.to_parquet(os.path.join(OUTPUT_DIR, "X_train.parquet"))
    X_test.to_parquet(os.path.join(OUTPUT_DIR, "X_test.parquet"))

    # Wrap y Series as DataFrames before saving
    y_train_df = y_train.to_frame(name="Price")
    y_test_df = y_test.to_frame(name="Price")

    y_train_df.to_parquet(os.path.join(OUTPUT_DIR, "y_train.parquet"))
    y_test_df.to_parquet(os.path.join(OUTPUT_DIR, "y_test.parquet"))

    print("Train/test split created.")

if __name__ == "__main__":
    main()