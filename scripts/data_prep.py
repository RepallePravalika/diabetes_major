# scripts/data_prep.py
import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

def load_data(path):
    df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def basic_cleaning(df):
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"🧹 Duplicates removed. Remaining rows: {len(df)}")
    return df

def handle_outliers(df, numeric_cols):
    for c in numeric_cols:
        low = df[c].quantile(0.01)
        high = df[c].quantile(0.99)
        df[c] = df[c].clip(lower=low, upper=high)
    return df

def preprocess(df, target_col='Outcome', test_size=0.2, random_state=42, k_features=8):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy='median')
    X_num = pd.DataFrame(imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

    X_num = handle_outliers(X_num, numeric_cols)
    selector = SelectKBest(score_func=f_classif, k=min(k_features, X_num.shape[1]))
    selector.fit(X_num, y)
    selected_features = X_num.columns[selector.get_support()].tolist()
    X_sel = X_num[selected_features]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_sel), columns=selected_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"📊 Train size: {X_train.shape}, Test size: {X_test.shape}")

    print("🔄 Applying SMOTE...")
    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"✅ After SMOTE: {X_train_res.shape[0]} samples")

    artifacts = {
        'X_train': X_train_res,
        'X_test': X_test,
        'y_train': y_train_res,
        'y_test': y_test,
        'scaler': scaler,
        'imputer': imputer,
        'selected_features': selected_features
    }
    return artifacts

def save_artifacts(artifacts, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for key in artifacts:
        joblib.dump(artifacts[key], os.path.join(out_dir, f'{key}.pkl'))
    print(f"💾 Artifacts saved to: {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV dataset')
    parser.add_argument('--out', default='../models', help='Output directory')
    parser.add_argument('--target', default='Outcome', help='Target column')
    args = parser.parse_args()

    df = load_data(args.data)
    df = basic_cleaning(df)
    artifacts = preprocess(df, target_col=args.target)
    save_artifacts(artifacts, args.out)
    print("✅ Preprocessing complete with SMOTE.")

if __name__ == '__main__':
    main()
