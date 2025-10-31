# scripts/evaluate.py
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    model = joblib.load('models/best_model.pkl')
    X_test = joblib.load('models/X_test.pkl')
    y_test = joblib.load('models/y_test.pkl')

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("📈 Evaluation Metrics:")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall   : {recall_score(y_test, preds):.4f}")
    print(f"F1 Score : {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")

if __name__ == '__main__':
    main()
