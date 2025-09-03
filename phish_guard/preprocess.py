import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources (first run only)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    # Lowercase
    text = text.lower()

    # Replace URLs with "url"
    text = re.sub(r"http\S+|www\S+", " url ", text)

    # Replace emails with "email"
    text = re.sub(r"\S+@\S+", " email ", text)

    # Replace numbers with "number"
    text = re.sub(r"\d+", " number ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    tokens = text.split()

    # Remove stopwords + Lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]

    return " ".join(cleaned_tokens)


# === Demo ===
if __name__ == "__main__":
    samples = [
        "Your account is suspended, click here: https://scam.com/login",
        "Contact us at support@example.com ASAP!!!",
        "Win $1000 NOW!!!",
        "Meeting at 3 PM tomorrow :)",
    ]

    for s in samples:
        print(f"Raw: {s}")
        print(f"Cleaned: {clean_text(s)}\n")
