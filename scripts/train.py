# scripts/train.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_models(X_train, y_train):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500),
        'DecisionTree': DecisionTreeClassifier(),
        'SVM': SVC(probability=True),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    }

    trained = {}
    for name, model in models.items():
        print(f"🔹 Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"✅ {name} done.")

    voting = VotingClassifier(
        estimators=[(k, v) for k, v in trained.items()],
        voting='soft'
    )
    voting.fit(X_train, y_train)
    trained['VotingClassifier'] = voting
    print("✅ Hybrid Voting Classifier trained.")
    return trained

def evaluate(models, X_test, y_test):
    print("\n📊 Model Evaluation:")
    for name, model in models.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name}: Accuracy = {acc:.4f}")
    print("\nVoting Classifier Report:")
    print(classification_report(y_test, models['VotingClassifier'].predict(X_test)))

def main():
    X_train = joblib.load('models/X_train.pkl')
    y_train = joblib.load('models/y_train.pkl')
    X_test = joblib.load('models/X_test.pkl')
    y_test = joblib.load('models/y_test.pkl')

    models = train_models(X_train, y_train)
    evaluate(models, X_test, y_test)
    joblib.dump(models['VotingClassifier'], 'models/best_model.pkl')
    print("💾 Saved best model: models/best_model.pkl")

if __name__ == '__main__':
    main()
