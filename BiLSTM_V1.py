# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:50:52 2025

@author: mariya
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 22:13:03 2025

@author: mariya
"""
from email import message_from_string, message_from_bytes, policy
from email.policy import default
import re
from urllib.parse import urlparse
from flask import Flask, request, render_template, redirect, url_for, flash, send_file, jsonify
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
import csv
import os
import hashlib
import requests
import time
import tldextract
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model("bilstm_url_model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Constants
MAX_LENGTH = 100
THRESHOLD = 40
WHITELISTED_DOMAINS = [
    "google.com", "facebook.com", "twitter.com", "linkedin.com", "github.com",
    "amazon.com", "microsoft.com", "apple.com", "instagram.com", "youtube.com",
    "reddit.com", "wikipedia.org", "yahoo.com", "dropbox.com", "zoom.us",
    "spotify.com", "pinterest.com", "tiktok.com", "quora.com", "medium.com",
    "slack.com", "vimeo.com", "canva.com", "airbnb.com", "paypal.com",
    "etsy.com", "snapchat.com", "soundcloud.com", "wordpress.com", "hulu.com",
    "disneyplus.com", "netflix.com", "whatsapp.com", "telegram.org", "bitbucket.org",
    "gitlab.com", "stackoverflow.com", "yandex.com", "shopify.com", "wix.com",
    "bestbuy.com", "target.com", "walmart.com", "macys.com"
]

scaler = MinMaxScaler(feature_range=(0, 1))
PREDICTIONS_FILE = "predictions.csv"

# Ensure predictions file exists
if not os.path.exists(PREDICTIONS_FILE):
    with open(PREDICTIONS_FILE, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["URL", "Label", "Confidence"])

# Feature extraction function
def extract_features(url):
    parsed_url = tldextract.extract(url)
    domain = f"{parsed_url.domain}.{parsed_url.suffix}"
    raw_features = [
        len(url), len(parsed_url.domain), len(parsed_url.suffix), len(parsed_url.subdomain),
        url.count('.'), url.count('-'), url.count('/'), url.count('='), url.count('?'),
        url.count('@'), url.count('%'), 1 if "https" in url.lower() else 0
    ]
    scaled_features = scaler.fit_transform([raw_features])
    return domain, scaled_features

# Preprocess URL function
def preprocess_url(url):
    sequence = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post")
    return padded

# Prediction function
def predict_url(url):
    domain, feature_input = extract_features(url)

    if domain in WHITELISTED_DOMAINS:
        label = "Legitimate"
        confidence = 100.0
    else:
        url_input = preprocess_url(url)
        prediction = model.predict([url_input, feature_input])
        confidence = prediction[0][0] * 100
        label = "Phishing" if confidence >= THRESHOLD else "Legitimate"

    with open(PREDICTIONS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([url, label, f"{confidence:.2f}%"])

    return label, confidence

API_KEY = "4ef4a2357d6638c1520fa4f228c24436154fd5a5e328cac50a99beae9bccebbe"
headers = {"x-apikey": API_KEY}

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def get_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def upload_to_virustotal(file_path):
    headers = {"x-apikey": API_KEY}
    with open(file_path, "rb") as f:
        response = requests.post(
            "https://www.virustotal.com/api/v3/files",
            headers=headers,
            files={"file": (os.path.basename(file_path), f)}
        )

    if response.status_code == 200:
        file_id = response.json()["data"]["id"]
        file_hash = get_file_hash(file_path)
        os.makedirs("temp_results", exist_ok=True)
        with open(f"temp_results/{file_id}.txt", "w") as f:
            f.write(file_hash)
        return file_id
    else:
        print(f"Upload failed: {response.status_code}")
        return None

def submit_and_get_file_id(file_path):
    file_hash = get_file_hash(file_path)
    headers = {"x-apikey": API_KEY}
    check_url = f"https://www.virustotal.com/api/v3/files/{file_hash}"

    response = requests.get(check_url, headers=headers)

    if response.status_code == 200:
        return response.json()["data"]["id"]
    elif response.status_code == 404:
        with open(file_path, "rb") as file:
            upload = requests.post(
                "https://www.virustotal.com/api/v3/files",
                headers=headers,
                files={"file": (os.path.basename(file_path), file)}
            )
        if upload.status_code == 200:
            return upload.json()["data"]["id"]
        else:
            raise Exception(f"Upload failed: {upload.status_code}")
    else:
        raise Exception(f"VirusTotal error: {response.status_code}")

@app.route("/check_analysis_status/<file_id>")
def check_analysis_status(file_id):
    headers = {"x-apikey": API_KEY}
    url = f"https://www.virustotal.com/api/v3/analyses/{file_id}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return jsonify({"status": "error"})

    data = response.json()
    status = data["data"]["attributes"]["status"]
    return jsonify({"status": status})

@app.route("/html_check", methods=["GET", "POST"])
def html_check():
    file_id = None

    if request.method == "POST":
        file = request.files.get("html_file")
        html_code = request.form.get("html_code")

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", filename)
            file.save(file_path)
            file_id = upload_to_virustotal(file_path)

        elif html_code:
            hashcode = hashlib.md5(html_code.encode()).hexdigest()
            temp_path = os.path.join("uploads", f"temp_{hashcode}.html")
            with open(temp_path, "w") as f:
                f.write(html_code)
            file_id = upload_to_virustotal(temp_path)

    return render_template("html_check.html", file_id=file_id)

@app.route("/html_report/<file_id>")
def html_report(file_id):
    headers = {"x-apikey": API_KEY}
    hash_file = f"temp_results/{file_id}.txt"
    if not os.path.exists(hash_file):
        return "Analysis hash not found."

    with open(hash_file, "r") as f:
        file_hash = f.read()

    response = requests.get(f"https://www.virustotal.com/api/v3/files/{file_hash}", headers=headers)
    if response.status_code == 200:
        stats = response.json()["data"]["attributes"]["last_analysis_stats"]
        return render_template("html_report.html",
                               harmless=stats["harmless"],
                               suspicious=stats["suspicious"],
                               malicious=stats["malicious"],
                               undetected=stats["undetected"])
    else:
        return "Error fetching report."

@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("dashboard"))

@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")

@app.route("/url_check", methods=["GET", "POST"])
def url_check():
    return render_template("url_check.html")

@app.route("/predict", methods=["POST"])
def predict():
    url = request.form["url"]
    label, confidence = predict_url(url)
    return render_template("predict.html", url=url, label=label, confidence=confidence)

@app.route("/generate_report", methods=["GET"])
def generate_report():
    predictions = []
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            predictions = list(reader)

    return render_template("generate_report.html", predictions=predictions)

@app.route("/delete_report", methods=["POST"])
def delete_report():
    with open(PREDICTIONS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["URL", "Label", "Confidence"])
    return redirect(url_for("generate_report"))

@app.route("/download_report", methods=["GET"])
def download_report():
    if os.path.exists(PREDICTIONS_FILE):
        return send_file(PREDICTIONS_FILE, as_attachment=True)
    return redirect(url_for("generate_report"))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
