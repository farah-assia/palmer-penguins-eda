"""
Script to train and save the production penguin classification pipeline.
Outputs: models/penguin_model.joblib
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

def train_and_export_model():
    data_path = Path("data/penguins.csv")
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "penguin_model.joblib"

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Cleaning: Drop rows with missing measurements (2 rows)
    numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    df_clean = df.dropna(subset=numeric_cols).copy()
    print(f"Cleaned dataset shape: {df_clean.shape} (342 observations)")

    # 2. Feature engineering: bill_ratio
    df_clean['bill_ratio'] = df_clean['bill_length_mm'] / df_clean['bill_depth_mm']

    # 3. Standardize sex values: uppercase or NaN
    df_clean['sex'] = df_clean['sex'].astype(str).str.upper().replace({'NAN': np.nan, 'NONE': np.nan})

    # Features and target (strictly excluding 'island' to prevent shortcut learning)
    X = df_clean[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'bill_ratio', 'sex']]
    y = df_clean['species']

    # 4. Scikit-learn Preprocessing Pipeline
    num_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'bill_ratio']
    cat_features = ['sex']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    # 5. Full Pipeline with Logistic Regression
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(C=10.0, max_iter=1000, random_state=42))
    ])

    # 6. Evaluation with Repeated Stratified 5-Fold CV (50 splits)
    print("Evaluating pipeline with Repeated Stratified 5x10 CV...")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=rskf, scoring='f1_macro', n_jobs=1)
    acc_scores = cross_val_score(pipeline, X, y, cv=rskf, scoring='accuracy', n_jobs=1)
    print(f"CV Macro F1 : {scores.mean():.4f} +/- {scores.std():.4f}")
    print(f"CV Accuracy : {acc_scores.mean():.4f} +/- {acc_scores.std():.4f}")

    # 7. Fit on the complete cleaned dataset
    print("Fitting model on complete cleaned dataset...")
    pipeline.fit(X, y)

    # 8. Save with joblib
    print(f"Saving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    print(f"[SUCCESS] Model successfully saved to {model_path}")

    # 9. Verify reload and test prediction
    loaded_model = joblib.load(model_path)
    test_sample = pd.DataFrame([{
        "bill_length_mm": 50.0,
        "bill_depth_mm": 15.0,
        "flipper_length_mm": 220.0,
        "body_mass_g": 5200.0,
        "bill_ratio": 50.0 / 15.0,
        "sex": "MALE"
    }])
    pred = loaded_model.predict(test_sample)[0]
    proba = loaded_model.predict_proba(test_sample)[0]
    print(f"Test sample prediction: {pred} (probabilities: {dict(zip(loaded_model.classes_, proba.round(4)))})")

if __name__ == "__main__":
    train_and_export_model()
