import shap
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def main():
    # Load the preprocessed dataset
    data = pd.read_csv("data/diabetes_synthetic_realistic_25000.csv")

    # Split into features and target
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Load trained hybrid model
    with open("models/final_hybrid_model.pkl", "rb") as f:
        model = pickle.load(f)

    print("✅ Model loaded successfully for SHAP explainability.")

    # Pick a base model for SHAP explanation (RandomForest is SHAP-compatible)
    rf_model = None
    for name, clf in model.estimators:
        if name == "RandomForest":
            rf_model = clf
            break

    if rf_model is None:
        print("⚠️ RandomForest not found in VotingClassifier. Using LogisticRegression instead.")
        rf_model = model.estimators_[0][1]

    # Use TreeExplainer for tree-based models
    explainer = shap.Explainer(rf_model, X_test)
    shap_values = explainer(X_test)

    print("✅ SHAP values generated successfully!")

    # Visualizations
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)
    shap.plots.bar(shap_values)

if __name__ == "__main__":
    main()
