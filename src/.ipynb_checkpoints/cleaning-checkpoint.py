import numpy as np
import pandas as pd

# ---- Minimal cleaning for v1 (RAW-ish) ----
def clean_v1_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """Keep v1 'raw' but drop rows with missing target so the model can train."""
    if 'deadlift' not in df.columns:
        raise ValueError("Column 'deadlift' not found in dataset.")
    return df[df['deadlift'].notna()].copy()

# ---- Your full cleaning for v2 ----
def clean_v2_full(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning you did in the notebook."""
    if 'deadlift' not in df.columns:
        raise ValueError("Column 'deadlift' not found in dataset.")

    data = df.copy()

    # Remove not relevant columns (drop rows with NA in required cols first)
    required = ['athlete_id', 'name', 'region', 'team', 'affiliate', 'gender', 'age',
                'height', 'weight', 'fran', 'helen', 'grace', 'filthy50', 'fgonebad',
                'run400', 'run5k', 'candj', 'snatch', 'deadlift', 'backsq', 'pullups',
                'eat', 'train', 'background', 'experience', 'schedule', 'howlong']
    existing_required = [c for c in required if c in data.columns]
    data = data.dropna(subset=existing_required)

    drop_cols = ['affiliate','team','name','athlete_id','fran','helen','grace',
                 'filthy50','fgonebad','run400','run5k','pullups','train']
    drop_cols = [c for c in drop_cols if c in data.columns]
    if drop_cols:
        data = data.drop(columns=drop_cols)

    # Remove outliers / invalid values (only if columns exist)
    if 'weight' in data.columns:
        data = data[data['weight'] < 1500]
    if 'gender' in data.columns:
        data = data[data['gender'] != '--']
    if 'age' in data.columns:
        data = data[data['age'] >= 18]
    if 'height' in data.columns:
        data = data[(data['height'] < 96) & (data['height'] > 48)]

    # lift bounds
    if {'deadlift','gender'}.issubset(data.columns):
        data = data[((data['deadlift'] > 0) & (data['deadlift'] <= 1105)) |
                    ((data['gender'] == 'Female') & (data['deadlift'] <= 636))]
    if 'candj' in data.columns:
        data = data[(data['candj'] > 0) & (data['candj'] <= 395)]
    if 'snatch' in data.columns:
        data = data[(data['snatch'] > 0) & (data['snatch'] <= 496)]
    if 'backsq' in data.columns:
        data = data[(data['backsq'] > 0) & (data['backsq'] <= 1069)]

    # Clean survey fields
    decline_dict = {'Decline to answer|': np.nan}
    data = data.replace(decline_dict)

    survey_cols = [c for c in ['background','experience','schedule','howlong','eat'] if c in data.columns]
    if survey_cols:
        data = data.dropna(subset=survey_cols)

    # Ensure target has no NaNs
    data = data[data['deadlift'].notna()].copy()

    return data