# Step 1: Import libraries
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import re
import string

# Clean text function
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

# Step 2: Load the datasets
def load_and_combine_datasets(fake_news_file, real_news_file, max_articles=20000):
    # Load the fake and real news datasets
    fake_news = pd.read_csv(fake_news_file)
    real_news = pd.read_csv(real_news_file)

    # Apply text cleaning
    fake_news['text'] = fake_news['text'].apply(clean_text)
    real_news['text'] = real_news['text'].apply(clean_text)

    # Label the datasets (1 for fake news, 0 for real news)
    fake_news['label'] = 1
    real_news['label'] = 0

    # Combine both datasets
    combined_data = pd.concat([fake_news, real_news])

    # Limit to the first `max_articles` articles
    combined_data = combined_data.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset
    combined_data = combined_data[:max_articles]  # Limit to the first max_articles

    return combined_data

# Step 3: Tokenize the dataset
def tokenize_data(combined_data):
    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(combined_data)

    # Load the DistilBERT tokenizer (lighter version of BERT)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Tokenize the text data
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Step 4: Split the dataset into train and test sets
def split_dataset(tokenized_dataset, test_size=0.2):
    train_test_split = tokenized_dataset.train_test_split(test_size=test_size)
    return train_test_split['train'], train_test_split['test']

# Step 5: Fine-tune the DistilBERT model (faster, smaller)
def train_bert_model(train_dataset, test_dataset, output_dir='./bert_fine_tuned'):
    # Load the pretrained DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # Updated from evaluation_strategy
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,  # Reduced epochs for faster training
        weight_decay=0.01,
        logging_dir='./logs',  # Directory for logging
        logging_steps=10,  # Log progress frequently
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

# Step 6: Main execution function
if __name__ == "__main__":
    # File paths for your datasets (Update to the correct filenames)
    fake_news_file = 'Fake.csv'
    real_news_file = 'True.csv'

    # Load and combine the datasets with a maximum of 20,000 articles
    combined_data = load_and_combine_datasets(fake_news_file, real_news_file, max_articles=20000)

    # Tokenize the dataset
    tokenized_dataset = tokenize_data(combined_data)

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = split_dataset(tokenized_dataset)

    # Train the DistilBERT model
    train_bert_model(train_dataset, test_dataset)
