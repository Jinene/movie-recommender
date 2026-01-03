import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """
    Load movie and rating datasets.
    Returns:
        movies: DataFrame of movies with features
        ratings: DataFrame of user ratings
    """
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    # Combine features (for now just genres)
    movies['features'] = movies['Genres']

    return movies, ratings

def recommend(title, movies):
    """
    Recommend top 10 movies similar to the input movie title.
    Args:
        title: str, movie title
        movies: DataFrame, movie data
    Returns:
        list of recommended movie titles
    """
    # Convert features to numeric vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['features'])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Find the index of the input movie
    idx = movies[movies['Title'] == title].index[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # top 10 similar movies

    movie_indices = [i[0] for i in sim_scores]
    return movies['Title'].iloc[movie_indices].tolist()
