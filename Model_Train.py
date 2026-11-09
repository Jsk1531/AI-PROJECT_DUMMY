
import numpy as np
import pandas as pd

def add_features(df):
    df["horsepower"] = (
        df["engine"].str.extract(r"(\d+(?:\.\d+)?)\s*HP", expand=False).astype(float)
    )

    # 2. Extract Engine Displacement in Liters (e.g., '3.7L' or '3.5 Liter' -> 3.7)
    df["displacement_l"] = (
        df["engine"]
        .str.extract(r"(\d+(?:\.\d+)?)\s*(?:L\b|Liter)", expand=False)
        .astype(float)
    )

    # 3. Extract Cylinder Count (e.g., 'V6', 'I4', 'W12', '4 Cylinder' -> 6, 4, 12)
    cyl_match = df["engine"].str.extract(
        r"(?:[VIW](\d{1,2})|(\d{1,2})\s*[- ]?[Cc]ylinder)", expand=True
    )
    df["cylinders"] = cyl_match[0].combine_first(cyl_match[1]).astype(float)

    # 4. Engine Type Flags (0 or 1)
    df["is_turbo"] = (
        df["engine"].str.contains(r"turbo", case=False, na=False).astype(int)
    )
    df["is_electric"] = (
        df["engine"].str.contains(r"electric", case=False, na=False).astype(int)
    )
    # Groupwise median imputation preserving brand/model hierarchy
    df["horsepower"] = df.groupby(["brand", "model"])["horsepower"].transform(
        lambda x: x.fillna(x.median())
    )
    df["displacement_l"] = df.groupby(["brand", "model"])[
        "displacement_l"
    ].transform(lambda x: x.fillna(x.median()))
    df["cylinders"] = df.groupby(["brand", "model"])["cylinders"].transform(
        lambda x: x.fillna(x.median())
    )

    # Fallback for unique models: fill with overall brand median or 0 for electric
    df["displacement_l"] = np.where(
        df["is_electric"] == 1, 0.0, df["displacement_l"].fillna(df["displacement_l"].median())
    )
    df["cylinders"] = np.where(
        df["is_electric"] == 1, 0.0, df["cylinders"].fillna(df["cylinders"].median())
    )
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
    brand_price_map = df.groupby("brand")["price"].median()
    df["brand_value"] = df["brand"].map(brand_price_map)

    clear_df = df[(df["price"] >= 1000) & (df["price"] <= 250000)].copy()


    features = [
        "brand_value",
        "model_year",
        "milage",
        "horsepower",
        "displacement_l",
        "cylinders",
        "is_turbo",
        "is_electric",
    ]

    X = clear_df[features]
    y = clear_df["price"]
    return df,X,y