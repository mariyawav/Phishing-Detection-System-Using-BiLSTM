import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tldextract
import re
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = tf.keras.models.load_model("bilstm_url_model.h5")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Define constants
MAX_LENGTH = 100  # Must match training setup
THRESHOLD = 40  # New phishing threshold (lowered)

# Whitelisted domains
WHITELISTED_DOMAINS = {
    "google.com", "facebook.com", "microsoft.com", "paypal.com",
    "apple.com", "amazon.com", "linkedin.com", "twitter.com",
    "instagram.com", "github.com", "wikipedia.org", "yahoo.com",
    "netflix.com", "bankofamerica.com", "chase.com"
}

# Initialize feature scaler
scaler = MinMaxScaler(feature_range=(0, 1))  # Ensure proper scaling

# Feature extraction function (Returns exactly 12 features)
def extract_features(url):
    """Extracts and scales 12 numerical features from a URL."""
    parsed_url = tldextract.extract(url)
    domain = f"{parsed_url.domain}.{parsed_url.suffix}"

    raw_features = [
        len(url),                         # 1. URL length
        len(parsed_url.domain),           # 2. Domain length
        len(parsed_url.suffix),           # 3. Suffix length
        len(parsed_url.subdomain),        # 4. Subdomain length
        url.count('.'),                   # 5. Number of dots
        url.count('-'),                   # 6. Number of hyphens
        url.count('/'),                   # 7. Number of slashes
        url.count('='),                   # 8. Number of equal signs
        url.count('?'),                   # 9. Number of question marks
        url.count('@'),                   # 10. Number of '@' symbols
        url.count('%'),                   # 11. Number of '%' symbols
        1 if "https" in url.lower() else 0 # 12. Is HTTPS? (Binary)
    ]

    # Scale features to match model input
    scaled_features = scaler.fit_transform([raw_features])

    return domain, scaled_features  # Ensure shape (1, 12)

# Preprocess URL function
def preprocess_url(url):
    """Tokenizes and pads the URL."""
    sequence = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post")
    return padded

# Prediction function
def predict_url(url):
    """Predicts whether a URL is phishing or legitimate."""
    domain, feature_input = extract_features(url)

    # Check if the domain is whitelisted
    if domain in WHITELISTED_DOMAINS:
        print(f"✅ Whitelisted: {url} (Legitimate)")
        return

    url_input = preprocess_url(url)

    # Make prediction
    prediction = model.predict([url_input, feature_input])
    confidence = prediction[0][0] * 100
    label = "Phishing" if confidence >= THRESHOLD else "Legitimate"

    print(f"🔍 URL: {url}")
    print(f"📌 Prediction: {label} ({confidence:.2f}% confidence)")

# Example Predictions
urls_to_test = [
    "facebook.com/login",  # Legitimate (Whitelisted)
    "secure-paypal.com/login",  # Phishing (Should now be detected)
    "microsoft.com/support",  # Legitimate (Whitelisted)
    "free-gift-card.com",  # Phishing (Should now be detected)
    "amazon.com/deals",  # Legitimate (Whitelisted)
    "paypal-security.com/login",  # Phishing (Should now be detected)
    "instagram.com/login",
    "amazon-verification.com" ,
    "appleid-security.com" ,
    "https://www.paypal.com/login"
]

for url in urls_to_test:
    predict_url(url)
