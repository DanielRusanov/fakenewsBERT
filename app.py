# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import string
import re
import os
from flask import Flask, request, jsonify, render_template
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Flask app initialization
app = Flask(__name__)

# Function to clean the text
def clean_text(text):
    text = re.sub(r'\b[A-Z]{2,}\s*\(Reuters\)|\(Reuters\)', '', text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  
    text = re.sub(r'\W+', ' ', text) 
    text = re.sub(r'http[s]?://\S+', '', text) 
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

# Load pre-trained Stacked model and vectorizer
try:
    best_stacked_model = joblib.load('best_stacking_model.pkl')
    vectorization = joblib.load('vectorizer.pkl')
except Exception as e:
    print(f"Error loading Stacked model or vectorizer: {str(e)}")

# Load pre-trained BERT model and tokenizer
try:
    bert_model = DistilBertForSequenceClassification.from_pretrained('./bert_fine_tuned')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
except Exception as e:
    print(f"Error loading BERT model: {str(e)}")

# Function for prediction using the Stacked model
def manual_testing(news):
    try:
        cleaned_text = clean_text(news)
        vectorized_text = vectorization.transform([cleaned_text])
        proba_stacked = best_stacked_model.predict_proba(vectorized_text)
        fake_prob = proba_stacked[0][0] * 100  # Probability of being Fake
        real_prob = proba_stacked[0][1] * 100  # Probability of being Real
        return {"Fake Probability": fake_prob, "Real Probability": real_prob}
    except Exception as e:
        print(f"Error in stacked model prediction: {str(e)}")
        return None

# Function for prediction using BERT model
def predict_news_bert(news_text):
    try:
        inputs = tokenizer(news_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()

            # Assuming 1 for Fake, 0 for Real
            if prediction == 1:
                fake_score = logits[0][1].item() * 100
                real_score = 100 - fake_score
                return "Fake", real_score, fake_score
            else:
                real_score = logits[0][0].item() * 100
                fake_score = 100 - real_score
                return "Real", real_score, fake_score
    except Exception as e:
        print(f"Error in BERT model prediction: {str(e)}")
        return None

# Route to serve the homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/work')
def work():
    return render_template('work.html')

# Route for Stacked model predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        news = request.form.get('news') or request.json.get('news')
        if not news:
            return jsonify({"error": "No news text provided"}), 400
        result = manual_testing(news)
        if result:
            return jsonify({
                "prediction": "Real" if result['Real Probability'] > result['Fake Probability'] else "Fake",
                "fake_score": result['Fake Probability'],
                "real_score": result['Real Probability']
            })
        else:
            return jsonify({"error": "Error in Stacked model prediction"}), 500
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Route for BERT model predictions
@app.route('/bert-predict', methods=['POST'])
def bert_predict():
    try:
        data = request.get_json()  # Get JSON data from the request
        news_text = data.get('news')

        if not news_text:
            return jsonify({'error': 'No input text provided'}), 400

        # Predict using the BERT model
        prediction, real_score, fake_score = predict_news_bert(news_text)

        return jsonify({
            'prediction': prediction,
            'real_score': real_score,
            'fake_score': fake_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for serving the BERT page
@app.route('/bert')
def bert_page():
    return render_template('bert.html')

if __name__ == '__main__':
    app.run(debug=True)
