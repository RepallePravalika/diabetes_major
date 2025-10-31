# scripts/chatbot.py
import joblib
import numpy as np

def chatbot():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/selected_features.pkl')

    print("\n🤖 Welcome to Diabetes Detection Chatbot!")
    print("Enter your health details to predict (type 'exit' to stop).")

    while True:
        user_input = input("\nType 'predict' to start or 'exit' to quit: ").strip().lower()
        if user_input == 'exit':
            break
        elif user_input == 'predict':
            values = []
            for f in features:
                val = float(input(f"Enter {f}: "))
                values.append(val)

            X = scaler.transform([values])
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1]
            print(f"🔹 Prediction: {'Diabetic' if pred == 1 else 'Non-Diabetic'} (Prob: {prob:.2f})")

if __name__ == '__main__':
    chatbot()
