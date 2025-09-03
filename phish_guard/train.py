import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from preprocess import clean_text   # 👈 Import cleaner

# === Load dataset ===
file_path = "data/sample.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Dataset file not found at: {file_path}")

df = pd.read_csv(file_path)

# === Basic checks ===
print("\n📂 Dataset Preview:")
print(df.head())
print(f"✅ Total samples: {len(df)}")

if len(df) < 2:
    raise ValueError("❌ Not enough samples in dataset. Please add more data!")

# === Apply preprocessing to text ===
df["clean_text"] = df["text"].apply(clean_text)

# === Features & Labels ===
X = df["clean_text"]
y = df["label"]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# === Text Vectorization ===
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === Model Training ===
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# === Evaluation ===
y_pred = model.predict(X_test_vec)
print("\n📈 Model Evaluation:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# === Save model & vectorizer ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/phish_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n💾 Model & vectorizer saved in 'models/' folder!")
