import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_recommender():
    print("Training Recommender Model...")

    # Load cleaned movies
    movies_path = "data/processed/movies_clean.csv"
    if not os.path.exists(movies_path):
        print(f"Error: {movies_path} not found. Ensure you have run preprocess.py block.")
        return

    movies = pd.read_csv(movies_path)

    # 1. Content-Based Filtering Setup
    # Prepare genres by removing the divider pipe
    print("Vectorizing genres...")
    movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)
    
    cv = CountVectorizer(stop_words='english')
    genre_matrix = cv.fit_transform(movies["genres_clean"])

    # Package the model elements
    model_data = {
        "movies": movies[["movieId", "title", "genres"]],
        "vectorizer": cv,
        "genre_matrix": genre_matrix
    }

    # Save logic
    os.makedirs("models", exist_ok=True)
    model_out = "models/recommender_model.pkl"
    with open(model_out, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Recommender model successfully saved to {model_out}")
    print("Testing functionality...")

    # Load and test
    with open(model_out, "rb") as f:
        loaded_model = pickle.load(f)
    print(f"Loaded {len(loaded_model['movies'])} movies into model!")

if __name__ == "__main__":
    train_recommender()
