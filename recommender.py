import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Fill missing values
movies["genres"] = movies["genres"].fillna("")

# Convert genres into numerical vectors
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create title index mapping
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()


def recommend(movie_title):
    if movie_title not in indices:
        return []

    idx = indices[movie_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return movies["title"].iloc[movie_indices].tolist()


def get_movie_titles():
    return sorted(movies["title"].tolist())