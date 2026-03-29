import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 6)

df = pd.read_csv("D:/Entertainment_AI_Project/Notebooks/data/processed/movie_data_filtered.csv")
df.head()

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

#Summary statistics
df.describe(include="all")

# Plot size is set to make the graph readable
plt.figure(figsize=(8,5))

# countplot shows how many times each rating value appears in the dataset
# This helps us understand the overall distribution of user ratings
sns.countplot(x="rating", data=df)

# Title and axis labels make the graph presentation-ready and easier to interpret
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")

# Rotating x-axis labels slightly improves readability if labels are crowded
plt.xticks(rotation=45)

# Display the plot
plt.show()

# Group by movie title and count how many ratings each movie received
# This tells us which movies are the most interacted with by users
most_rated = df.groupby("title")["rating"].count().sort_values(ascending=False).head(10)

most_rated


# Bar chart is useful here because we want to compare counts across top movies
most_rated.plot(kind="bar")

plt.title("Top 10 Most Rated Movies")
plt.xlabel("Movie Title")
plt.ylabel("Number of Ratings")

# Movie names are long, so rotation improves readability
plt.xticks(rotation=75)

plt.show()

# We calculate both average rating and number of ratings
# because a movie with only 2 ratings should not be ranked above a movie with 500 ratings
movie_stats = df.groupby("title").agg(
    avg_rating=("rating", "mean"),
    num_ratings=("rating", "count")
).reset_index()

# Apply a minimum rating count filter to make the ranking more reliable
top_rated = movie_stats[movie_stats["num_ratings"] >= 100].sort_values(
    by="avg_rating", ascending=False
).head(10)

top_rated

# Horizontal bar plot works better when movie titles are long
sns.barplot(data=top_rated, x="avg_rating", y="title")

plt.title("Top 10 Highest Rated Movies (min 100 ratings)")
plt.xlabel("Average Rating")
plt.ylabel("Movie Title")

plt.show()

# Count how many ratings each user has given
# This helps us understand whether a few users are contributing a lot of the data
active_users = df["userId"].value_counts().head(10)

active_users

active_users.plot(kind="bar")

plt.title("Top 10 Most Active Users")
plt.xlabel("User ID")
plt.ylabel("Number of Ratings Given")
plt.xticks(rotation=45)

plt.show()

# value_counts gives the number of ratings per user
user_activity = df["userId"].value_counts()

plt.figure(figsize=(8,5))

# Histogram helps us see how user activity is spread across the dataset
sns.histplot(user_activity, bins=50, kde=True)

plt.title("Distribution of User Activity")
plt.xlabel("Number of Ratings per User")
plt.ylabel("Frequency")

plt.show()

# Genres are stored together in one string separated by '|'
# explode converts multi-genre entries into separate rows for counting
genres_split = df["genres"].str.split("|").explode()

genre_counts = genres_split.value_counts().head(15)

genre_counts

genre_counts.plot(kind="bar")

plt.title("Top 15 Most Common Genres")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.show()

# First compute average rating per movie
# This reduces repeated rows because the same movie appears many times in the dataset
movie_avg = df.groupby(["movieId", "title", "genres"], as_index=False)["rating"].mean()
movie_avg.rename(columns={"rating": "avg_rating"}, inplace=True)

# Split genre strings into lists
movie_avg["genre"] = movie_avg["genres"].str.split("|")

# explode is now much safer because we are working on unique movies, not all rating rows
genre_movie_df = movie_avg.explode("genre")

# Compute average rating for each genre
genre_avg_rating = genre_movie_df.groupby("genre")["avg_rating"].mean().sort_values(ascending=False)

genre_avg_rating

genre_avg_rating.plot(kind="bar")

plt.title("Average Rating by Genre")
plt.xlabel("Genre")
plt.ylabel("Average Movie Rating")
plt.xticks(rotation=45)

plt.show()

# Use only unique movies so each movie is counted once
movies_unique = df[["movieId", "title", "genres", "year"]].drop_duplicates()

# Count how many unique movies belong to each release year
year_counts = movies_unique["year"].value_counts().sort_index()

year_counts.tail(20)

plt.figure(figsize=(12,5))

# Line plot helps show how the number of released movies changes over time
year_counts.plot()

plt.title("Number of Movies by Release Year")
plt.xlabel("Year")
plt.ylabel("Number of Movies")

plt.tight_layout()
plt.show()

# Group movie-wise and calculate average rating and total number of ratings
# This creates a compact movie-level summary from the large rating-level dataset
popularity_df = df.groupby(["movieId", "title"], as_index=False).agg(
    avg_rating=("rating", "mean"),
    num_ratings=("rating", "count")
)

popularity_df.head()

# Keep only movies with enough ratings so recommendations are reliable
# A movie with 5.0 rating from only 2 users should not rank at the top
popular_movies = popularity_df[popularity_df["num_ratings"] >= 100].copy()

# Sort by highest average rating, and use number of ratings as tie-breaker
popular_movies = popular_movies.sort_values(
    by=["avg_rating", "num_ratings"],
    ascending=[False, False]
)

popular_movies.head(10)

# Show the top 10 most recommendable popular movies
top_10_popular = popular_movies.head(10)
top_10_popular


# Visualize the top recommended popular movies
plt.figure(figsize=(10,6))
sns.barplot(data=top_10_popular, x="avg_rating", y="title")

plt.title("Top 10 Popular Movies Based on Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("Movie Title")

plt.tight_layout()
plt.show()

def get_popular_movies(top_n=10, min_ratings=100):
    # Filter movies with enough ratings
    filtered = popularity_df[popularity_df["num_ratings"] >= min_ratings].copy()

    # Rank movies by rating first, then by number of ratings
    filtered = filtered.sort_values(
        by=["avg_rating", "num_ratings"],
        ascending=[False, False]
    )

    return filtered.head(top_n)

get_popular_movies()

# Keep only one row per movie
movies_unique = df[["movieId", "title", "genres"]].drop_duplicates()

movies_unique.head()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Replace separator so CountVectorizer treats genres as separate tokens
movies_unique["genres_clean"] = movies_unique["genres"].str.replace("|", " ", regex=False)

movies_unique[["title", "genres", "genres_clean"]].head()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_unique = df[["movieId", "title", "genres"]].drop_duplicates().reset_index(drop=True)
movies_unique["genres_clean"] = movies_unique["genres"].str.replace("|", " ", regex=False)

cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies_unique["genres_clean"])

movie_indices = pd.Series(movies_unique.index, index=movies_unique["title"]).drop_duplicates()

def recommend_similar_movies(movie_title, top_n=10):
    # Check if movie exists
    if movie_title not in movie_indices:
        return f"Movie '{movie_title}' not found in dataset."

    # Get selected movie index
    idx = movie_indices[movie_title]

    # Compute similarity only for the selected movie, not the full dataset
    sim_scores = cosine_similarity(genre_matrix[idx], genre_matrix).flatten()

    # Get top similar movie indices
    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]

    return movies_unique[["title", "genres"]].iloc[similar_indices]

recommend_similar_movies("Shawshank Redemption, The (1994)")

# Count ratings per movie and per user
movie_rating_counts = df["title"].value_counts()
user_rating_counts = df["userId"].value_counts()

# Keep only popular movies and active users to reduce memory usage
selected_movies = movie_rating_counts[movie_rating_counts >= 500].index
selected_users = user_rating_counts[user_rating_counts >= 200].index

cf_df = df[
    (df["title"].isin(selected_movies)) &
    (df["userId"].isin(selected_users))
]

print("Collaborative filtering subset shape:", cf_df.shape)