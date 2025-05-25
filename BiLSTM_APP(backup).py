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

app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model("bilstm_url_model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Constants
MAX_LENGTH = 100
THRESHOLD = 40
WHITELISTED_DOMAINS = {"google.com", "facebook.com", "microsoft.com", "paypal.com",
    "apple.com", "amazon.com", "linkedin.com", "twitter.com",
    "instagram.com", "github.com", "wikipedia.org", "yahoo.com",
    "netflix.com", "bankofamerica.com", "chase.com"}

scaler = MinMaxScaler(feature_range=(0, 1))
PREDICTIONS_FILE = "predictions.csv"

# Ensure predictions file exists
if not os.path.exists(PREDICTIONS_FILE):
    with open(PREDICTIONS_FILE, "w", newline='') as f:
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
        confidence = 100.0  # Assume 100% confidence for whitelisted domains
    else:
        url_input = preprocess_url(url)
        prediction = model.predict([url_input, feature_input])
        confidence = prediction[0][0] * 100
        label = "Phishing" if confidence >= THRESHOLD else "Legitimate"

    # Save all URLs (both Legitimate & Phishing)
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
    with open(file_path, "rb") as f:
        response = requests.post(
            "https://www.virustotal.com/api/v3/files",
            headers=headers,
            files={"file": (os.path.basename(file_path), f)}
        )

    if response.status_code == 200:
        file_id = response.json()["data"]["id"]

        # Store file hash to retrieve report later
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
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            file_id = upload_to_virustotal(file_path)

        elif html_code:
            hashcode = hashlib.md5(html_code.encode()).hexdigest()
            temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{hashcode}.html")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(html_code)

            file_id = upload_to_virustotal(temp_path)

    return render_template("html_check.html", file_id=file_id)

@app.route("/html_report/<file_id>")
def html_report(file_id):
    # Load stored hash
    hash_file = f"temp_results/{file_id}.txt"
    if not os.path.exists(hash_file):
        return "Analysis hash not found."

    with open(hash_file, "r") as f:
        file_hash = f.read()

    response = requests.get(f"https://www.virustotal.com/api/v3/files/{file_hash}", headers=headers)
    if response.status_code == 200:
        stats = response.json()["data"]["attributes"]["last_analysis_stats"]
        return render_template("html_report.html",
                               harmless=stats.get("harmless", 0),
                               suspicious=stats.get("suspicious", 0),
                               malicious=stats.get("malicious", 0),
                               undetected=stats.get("undetected", 0))
    else:
        return "Error fetching report."

@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("dashboard"))  # Default to Dashboard

@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")  # Shows Dashboard

@app.route("/url_check", methods=["GET", "POST"])
def url_check():
    return render_template("url_check.html")  # Input form for checking URLs

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
            next(reader, None)  # Skip header if it exists
            predictions = list(reader)

    return render_template("generate_report.html", predictions=predictions)

@app.route("/delete_report", methods=["POST"])
def delete_report():
    with open(PREDICTIONS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["URL", "Label", "Confidence"])  # Recreate header
    
    return redirect(url_for("generate_report"))

@app.route("/download_report", methods=["GET"])
def download_report():
    if os.path.exists(PREDICTIONS_FILE):
        return send_file(PREDICTIONS_FILE, as_attachment=True)
    return redirect(url_for("generate_report"))

@app.route("/domainlookup", methods=["GET", "POST"])
def domainlookup():
    if request.method == "POST":
        domain = request.form.get("domain")
        return redirect(url_for("domain_report", domain=domain))
    return render_template("domainlookup.html")

# Different API key for domain lookup APIs
DOMAIN_API_KEY = "at_l8ukmntRGsHuJCi12IECxHgpHiWvP"
domain_headers = {"x-apikey": DOMAIN_API_KEY}

@app.route("/domain_report", methods=["GET"])
def domain_report():
    domain = request.args.get("domain")
    if not domain:
        return "No domain provided", 400

    # Define URLs
    apis = {
        "whois": f"https://www.whoisxmlapi.com/whoisserver/WhoisService?apiKey={DOMAIN_API_KEY}&domainName={domain}&outputFormat=JSON",
        "screenshot": f"https://website-screenshot.whoisxmlapi.com/api/v1?apiKey={DOMAIN_API_KEY}&domainName={domain}",
        "ip_geolocation": "",  # Will set after extracting IP
        "dns_lookup": f"https://dns-lookup.whoisxmlapi.com/api/v1?apiKey={DOMAIN_API_KEY}&domainName={domain}",
        "domain_reputation": f"https://domain-reputation.whoisxmlapi.com/api/v1?apiKey={DOMAIN_API_KEY}&domainName={domain}",
        "website_categorization": f"https://website-categorization.whoisxmlapi.com/api/v1?apiKey={DOMAIN_API_KEY}&domainName={domain}",
        "ssl_certificates": f"https://ssl-certificates.whoisxmlapi.com/api/v1?apiKey={DOMAIN_API_KEY}&domainName={domain}",
        "threat_intelligence": f"https://threat-intelligence.whoisxmlapi.com/api/v1?apiKey={DOMAIN_API_KEY}&domainName={domain}",
    }

    def get_api_data(url):
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                return resp.json()
            else:
                return {}
        except Exception:
            return {}

    # Get WHOIS to extract IP
    whois_data = get_api_data(apis["whois"])
    whois_info = whois_data.get("WhoisRecord", {})
    registrar = whois_info.get("registrarName", "N/A")
    creation_date = whois_info.get("createdDate", "N/A")
    updated_date = whois_info.get("updatedDate", "N/A")
    country = whois_info.get("registryData", {}).get("registrant", {}).get("country", "N/A")

    # Attempt to get IP address from registryData or fallback to default
    ips = whois_info.get("registryData", {}).get("ips", [])
    ip_address = ips[0] if ips else "8.8.8.8"

    # Update IP geolocation API URL
    apis["ip_geolocation"] = f"https://ip-geolocation.whoisxmlapi.com/api/v1?apiKey={DOMAIN_API_KEY}&ipAddress={ip_address}"

    # Fetch all API data
    screenshot_data = get_api_data(apis["screenshot"])
    ip_geolocation_data = get_api_data(apis["ip_geolocation"])
    dns_lookup_data = get_api_data(apis["dns_lookup"])
    domain_reputation_data = get_api_data(apis["domain_reputation"])
    website_categorization_data = get_api_data(apis["website_categorization"])
    ssl_certificates_data = get_api_data(apis["ssl_certificates"])
    threat_intelligence_data = get_api_data(apis["threat_intelligence"])

    # Process IP Geolocation
    ip_info = ip_geolocation_data.get("location", {})
    ip_details = {
        "ip": ip_address,
        "city": ip_info.get("city", "N/A"),
        "country": ip_info.get("country", "N/A"),
        "asn": ip_info.get("asn", "N/A"),
        "isp": ip_info.get("isp", "N/A")
    }

    # Process Reputation
    reputation = {
        "score": domain_reputation_data.get("reputationScore", "N/A"),
        "verdict": domain_reputation_data.get("verdict", "N/A"),
        "category": website_categorization_data.get("categories", ["N/A"])[0]
    }

    # Process DNS
    dns = {
        "mx": ", ".join([record.get("exchange", "") for record in dns_lookup_data.get("MX", [])]),
        "ns": ", ".join([record.get("name", "") for record in dns_lookup_data.get("NS", [])]),
        "a": ", ".join([record.get("ip", "") for record in dns_lookup_data.get("A", [])])
    }

    # SSL Cert Info
    ssl_certificates = ssl_certificates_data.get("result", {}).get("certificates", [])
    ssl = {}
    if ssl_certificates:
        ssl_data = ssl_certificates[0]
        ssl = {
            "issuer": ssl_data.get("issuer", {}).get("organizationName", "N/A"),
            "valid_from": ssl_data.get("validFrom", "N/A"),
            "valid_to": ssl_data.get("validTo", "N/A")
        }
    else:
        ssl = {"issuer": "N/A", "valid_from": "N/A", "valid_to": "N/A"}

    # Threat Intel
    threats = {
        "malware": threat_intelligence_data.get("malwareDetected", "N/A"),
        "phishing": threat_intelligence_data.get("phishingDetected", "N/A"),
        "urls": ", ".join(threat_intelligence_data.get("suspiciousURLs", [])) or "N/A"
    }

    # Screenshot URL
    screenshot_url = screenshot_data.get("screenshotUrl", "")

    # Final Report
    report = {
        "whois": {
            "registrar": registrar,
            "creation_date": creation_date,
            "updated_date": updated_date,
            "country": country
        },
        "ip": ip_details,
        "reputation": reputation,
        "dns": dns,
        "ssl": ssl,
        "threats": threats,
        "screenshot": screenshot_url
    }

    return render_template("domain_report.html", report=report)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
