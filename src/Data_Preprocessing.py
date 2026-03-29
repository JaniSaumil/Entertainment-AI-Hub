import pandas as pd
import os

# File paths
movies_path = "data/raw/movies.csv"
ratings_path = "data/raw/ratings.csv"

print("Loading datasets...")
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# Basic info
print("\nMovies shape:", movies.shape)
print("Ratings shape:", ratings.shape)

print("\nMovies columns:", movies.columns.tolist())
print("Ratings columns:", ratings.columns.tolist())

print("\nMissing values in movies:")
print(movies.isnull().sum())

print("\nMissing values in ratings:")
print(ratings.isnull().sum())

print("\nDuplicate rows in movies:", movies.duplicated().sum())
print("Duplicate rows in ratings:", ratings.duplicated().sum())

# Remove duplicates
movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)

# Handle missing values
movies.dropna(subset=["movieId", "title", "genres"], inplace=True)
ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"], inplace=True)

# Extract year from title
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
movies["year"] = pd.to_numeric(movies["year"], errors="coerce")

# Clean movie title
movies["clean_title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

# Convert timestamp
ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce")

# Keep valid ratings
ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5.0)]

print("\nMerging datasets...")
movie_data = pd.merge(ratings, movies, on="movieId", how="left")

print("Merged dataset shape:", movie_data.shape)

print("\nFiltering active users and popular movies...")
user_counts = movie_data["userId"].value_counts()
movie_counts = movie_data["movieId"].value_counts()

active_users = user_counts[user_counts >= 20].index
popular_movies = movie_counts[movie_counts >= 20].index

filtered_data = movie_data[
    (movie_data["userId"].isin(active_users)) &
    (movie_data["movieId"].isin(popular_movies))
]

print("Filtered dataset shape:", filtered_data.shape)

# Create output folder
os.makedirs("data/processed", exist_ok=True)

# Save outputs
movies.to_csv("data/processed/movies_clean.csv", index=False)
ratings.to_csv("data/processed/ratings_clean.csv", index=False)
movie_data.to_csv("data/processed/movie_data_merged.csv", index=False)
filtered_data.to_csv("data/processed/movie_data_filtered.csv", index=False)

print("\nFinal Summary")
print("Unique users:", filtered_data["userId"].nunique())
print("Unique movies:", filtered_data["movieId"].nunique())
print("Total ratings:", len(filtered_data))

print("\nSample rows:")
print(filtered_data.head())

print("\nPreprocessing complete.")