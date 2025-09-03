import joblib
import os
from preprocess import clean_text   # 👈 import preprocessing

# === Load model & vectorizer ===
model_path = "models/phish_model.pkl"
vectorizer_path = "models/vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("❌ Model or vectorizer not found! Please train first.")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# === Prediction function ===
def predict_message(message: str) -> str:
    # Clean message first
    cleaned = clean_text(message)
    # Vectorize
    vec = vectorizer.transform([cleaned])
    # Predict
    pred = model.predict(vec)[0]
    return pred

# === Demo ===
if __name__ == "__main__":
    samples = [
        "Your account is suspended, click here to verify",
        "Let's meet for coffee tomorrow",
        "Update your billing information immediately",
        "Happy birthday! Have a great day 🎉"
    ]
    for msg in samples:
        print(f"\n📩 Message: {msg}")
        print(f"🔎 Prediction: {predict_message(msg)}")
