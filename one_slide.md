# Data Versioning & Model Results (v1 vs v2 vs v2-DP)

**Tools:** lakeFS (design), DVC (implementation on GCS)

**Dataset v1 (raw-ish):** minimal clean (drop NaN target), baseline Linear Regression (impute + one-hot)  
**Metrics:** R²=-204.6834, MAE=13183.7903, RMSE=1120191.8318

**Dataset v2 (cleaned):** outlier bounds, survey NA removal, irrelevant cols dropped  
**Metrics:** R²=0.8399, MAE=27.9291, RMSE=35.6037

**DP model (v2, DP-SGD):** noise_multiplier=1.1, clip=1.0, epochs=10, batch=64, δ=1e-05  
**Metrics:** MAE=108.8913, RMSE=130.9931, **ε=5.3985**

**Takeaways:**  
- v2 vs v1: note MAE/RMSE/R² changes.  
- DP vs non-DP (v2): ε indicates privacy level; expect some accuracy trade-off.
