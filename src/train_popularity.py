import os
import pickle
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer

def train_popularity():
    print("Training Popularity Prediction Model using TMDB Dataset...")
    
    tmdb_path = "data/raw/tmdb_5000_movies.csv"
    
    if not os.path.exists(tmdb_path):
         print(f"Error: Missing dataset ({tmdb_path}). Please download it from Kaggle.")
         return
    
    df = pd.read_csv(tmdb_path)
    
    print(f"Loaded {len(df)} movies from TMDB.")
    
    # Preprocessing
    # 1. Handle Missing Values
    df = df.dropna(subset=['budget', 'runtime', 'popularity', 'genres'])
    df = df[(df['budget'] > 0) & (df['runtime'] > 0)] # Filter out bad/zero entries
    
    # 2. Parse JSON Genres
    def parse_genres(genre_str):
        try:
            # TMDB stores genres as JSON strings: '[{"id": 28, "name": "Action"}, ...]'
            genres = json.loads(genre_str)
            return [g['name'] for g in genres]
        except:
            return []

    print("Extracting features (Budget, Runtime, Genres)...")
    df['genre_list'] = df['genres'].apply(parse_genres)
    
    # Drop rows that have no genres
    df = df[df['genre_list'].map(len) > 0]
    
    # 3. Encode Genres
    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(mlb.fit_transform(df["genre_list"]), columns=mlb.classes_, index=df.index)
    
    # 4. Final Feature Matrix X and Target y
    X = pd.concat([df[['budget', 'runtime']], genre_encoded], axis=1)
    y = df['popularity']
    
    print(f"Training RandomForestRegressor on {len(X)} high-quality movies...")
    # More robust model for this dataset
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X, y)
    
    print("Popularity Model trained (R^2 on train data):", round(model.score(X, y), 3))
    
    os.makedirs("models", exist_ok=True)
    model_data = {
        "model": model,
        "mlb": mlb,
        "features": list(X.columns),
        "max_popularity": float(y.max()) # we keep max popularity to scale the UI properly
    }
    
    model_out = "models/popularity_model_tmdb.pkl"
    with open(model_out, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Popularity model successfully saved to {model_out}")

if __name__ == "__main__":
    train_popularity()
