import json

def quick_eda(df):
    """Simple exploratory data summary."""
    return {
        "shape": df.shape,
        "null_counts": df.isnull().sum().to_dict(),
        "describe_numeric": df.describe().to_dict()
    }

def save_report(report, path="eda_v1.json"):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)