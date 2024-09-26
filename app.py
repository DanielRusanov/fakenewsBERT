# Import  libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import string
import re
import os
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Flask app 
app = Flask(__name__)

# Function to clean the text
def clean_text(text):
    # Remove mentions of "Reuters", photo credits, etc.
    text = re.sub(r'\b[A-Z]{2,}\s*\(Reuters\)|\(Reuters\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Photo by.*?(Getty Images|AP|Reuters)\.', '', text, flags=re.IGNORECASE) 
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\[.*?\]', '', text)  
    text = re.sub(r'\W+', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Train and save the Stacking Model if it does not exist
def train_and_save_stacking_model():
    df_fake = pd.read_csv("Fake.csv")
    df_real = pd.read_csv("True.csv")

    # Add labels
    df_fake['label'] = 1  
    df_real['label'] = 0  

    # Sample 20,000 articles from each dataset to balance them
    df_fake_sample = df_fake.sample(n=20000, random_state=42, replace=False)
    df_real_sample = df_real.sample(n=20000, random_state=42, replace=False)

    # Combine and shuffle sampled datasets
    df = pd.concat([df_fake_sample, df_real_sample])
    df = df.sample(frac=1).reset_index(drop=True)  

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Define base models
    base_models = [
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())
    ]

    # Stacking classifier
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())

    # Train the stacking model
    stacking_model.fit(X_train_tfidf, y_train)

    # Save the trained model and vectorizer
    joblib.dump(stacking_model, 'best_stacking_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Stacking model and vectorizer saved.")


# load the Stacked model and vectorizer
try:
    best_stacked_model = joblib.load('best_stacking_model.pkl')
    vectorization = joblib.load('vectorizer.pkl')
except:
    print("Stacking model or vectorizer not found, training a new model...")
    train_and_save_stacking_model()
    best_stacked_model = joblib.load('best_stacking_model.pkl')
    vectorization = joblib.load('vectorizer.pkl')

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
        fake_prob = proba_stacked[0][0] * 100  
        real_prob = proba_stacked[0][1] * 100  
        return {"Fake Probability": fake_prob, "Real Probability": real_prob}
    except Exception as e:
        print(f"Error in stacked model prediction: {str(e)}")
        return None

# Function for prediction using BERT model
def predict_news_bert(news_text):
    try:
        cleaned_text = clean_text(news_text)  
        inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits

            # Apply softmax to convert logits to probabilities
            probabilities = F.softmax(logits, dim=-1)

            # Get the predicted class
            prediction = torch.argmax(probabilities, dim=-1).item()

            # Extract the probabilities for fake and real
            fake_score = probabilities[0][0].item() * 100  
            real_score = probabilities[0][1].item() * 100  

            # Return the result with proper scores
            return "Fake" if prediction == 0 else "Real", real_score, fake_score
    except Exception as e:
        print(f"Error in BERT model prediction: {str(e)}")
        return None

# Routes for pages
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
        data = request.get_json()  
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

# Route for serving the BERT model page
@app.route('/bert')
def bert_page():
    return render_template('bert.html')

if __name__ == '__main__':
    app.run(debug=True)
