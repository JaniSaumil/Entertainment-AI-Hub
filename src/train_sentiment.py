import os
import pickle
import pandas as pd
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def train_sentiment():
    print("Downloading NLTK Movie Reviews Dataset...")
    nltk.download('movie_reviews')
    
    print("Extracting reviews and labels...")
    documents = []
    labels = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append(movie_reviews.raw(fileid))
            labels.append(1 if category == 'pos' else 0)
    
    print(f"Loaded {len(documents)} reviews.")
    print("Training TF-IDF Logistic Regression Pipeline...")
    
    model = make_pipeline(
        TfidfVectorizer(stop_words='english', max_features=5000),
        LogisticRegression(max_iter=1000)
    )
    
    model.fit(documents, labels)
    print("Model Training Complete! Accuracy on train set:", round(model.score(documents, labels), 3))
    
    os.makedirs("models", exist_ok=True)
    model_out = "models/sentiment_model.pkl"
    with open(model_out, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Sentiment model successfully saved to {model_out}")

if __name__ == "__main__":
    train_sentiment()
