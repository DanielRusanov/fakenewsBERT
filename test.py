# Test the accuracy of the models #

# Import Libaries
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

# Load datasets
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# labels
df_fake['label'] = 1  
df_real['label'] = 0  

# Combine and shuffle datasets
df = pd.concat([df_fake, df_real])
df = df.sample(frac=1).reset_index(drop=True)  

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)



# Load the same TF-IDF vectorizer that was used during training
tfidf_vectorizer = joblib.load("vectorizer.pkl")  

# Load the trained stacking model
stacking_model = joblib.load("best_stacking_model.pkl")

# Transform the test set using the loaded TF-IDF vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)  

# Make predictions with the stacking classifier
stacking_predictions = stacking_model.predict(X_test_tfidf)

# Calculate metrics for the stacking model
stacking_accuracy = accuracy_score(y_test, stacking_predictions)
stacking_precision = precision_score(y_test, stacking_predictions)
stacking_recall = recall_score(y_test, stacking_predictions)
stacking_f1 = f1_score(y_test, stacking_predictions)

print(f"Stacking Classifier - Accuracy: {stacking_accuracy}")
print(f"Stacking Classifier - Precision: {stacking_precision}")
print(f"Stacking Classifier - Recall: {stacking_recall}")
print(f"Stacking Classifier - F1 Score: {stacking_f1}")

# BERT MODEL #

# Load the fine-tuned DistilBERT model and tokenizer
bert_model = DistilBertForSequenceClassification.from_pretrained('bert_fine_tuned')  
bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the test set for BERT
encoded_inputs = bert_tokenizer(list(X_test), return_tensors="pt", padding=True, truncation=True, max_length=512)

# Create a TensorDataset and DataLoader for batching
test_dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'])
test_loader = DataLoader(test_dataset, batch_size=4)  

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# Make predictions with BERT in batches
all_predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy()) 

# Calculate metrics for the BERT model
bert_accuracy = accuracy_score(y_test, all_predictions)
bert_precision = precision_score(y_test, all_predictions)
bert_recall = recall_score(y_test, all_predictions)
bert_f1 = f1_score(y_test, all_predictions)

print(f"BERT Model - Accuracy: {bert_accuracy}")
print(f"BERT Model - Precision: {bert_precision}")
print(f"BERT Model - Recall: {bert_recall}")
print(f"BERT Model - F1 Score: {bert_f1}")
