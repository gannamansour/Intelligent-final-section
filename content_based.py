"""
======================================================
 Hybrid Movie Recommendation System
 Script 2: Content-Based Filtering
         (TF-IDF + Cosine Similarity on Genres/Titles)
======================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFilter:
    """
    Content-Based Filtering using TF-IDF on movie genres + title keywords,
    combined with cosine similarity to find similar movies.
    """

    def __init__(self):
        self.vectorizer   = None
        self.tfidf_matrix = None
        self.movies_df    = None
        self.movie_indices = {}    # title → dataframe index
        self.id_to_idx    = {}     # movie_id → dataframe index

    # ─────────────────────────────────────────────
    # Feature Engineering
    # ─────────────────────────────────────────────

    def _build_content_string(self, row):
        """
        Combine genres and title words into a single text string for TF-IDF.
        Genres are repeated to give them more weight.
        """
        genres = row['genres'].replace('|', ' ').lower()
        # Extract words from title (remove year)
        title  = row['title'].split('(')[0].strip().lower()
        # Repeat genres twice for emphasis
        return f"{genres} {genres} {title}"

    def fit(self, movies_df):
        """Build TF-IDF matrix from movie metadata."""
        self.movies_df = movies_df.reset_index(drop=True).copy()

        # Build content strings
        self.movies_df['content'] = self.movies_df.apply(
            self._build_content_string, axis=1
        )

        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=1,
            stop_words='english'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.movies_df['content']
        )

        # Build lookup maps
        self.movie_indices = pd.Series(
            self.movies_df.index,
            index=self.movies_df['title']
        )
        self.id_to_idx = pd.Series(
            self.movies_df.index,
            index=self.movies_df['movie_id']
        )

        print("[ContentBased] TF-IDF matrix built.")
        print(f"  Shape : {self.tfidf_matrix.shape}")
        print(f"  Vocab : {len(self.vectorizer.vocabulary_):,} terms")

    # ─────────────────────────────────────────────
    # Similarity & Recommendations
    # ─────────────────────────────────────────────

    def get_similar_movies(self, movie_id, top_n=10):
        """
        Given a movie_id, return the top_n most similar movies
        based on cosine similarity of TF-IDF vectors.
        """
        if movie_id not in self.id_to_idx:
            return pd.DataFrame()

        idx = self.id_to_idx[movie_id]
        movie_vec = self.tfidf_matrix[idx]

        # Compute cosine similarity with all movies
        sim_scores = cosine_similarity(movie_vec, self.tfidf_matrix).flatten()

        # Exclude the movie itself
        sim_scores[idx] = -1

        top_indices = np.argsort(sim_scores)[::-1][:top_n]
        top_scores  = sim_scores[top_indices]

        result = self.movies_df.iloc[top_indices][['movie_id', 'title', 'genres']].copy()
        result['cb_similarity'] = top_scores
        return result.reset_index(drop=True)

    def recommend_for_user(self, user_id, ratings_df, top_n=10):
        """
        Generate content-based recommendations for a user.
        Strategy: find movies similar to the user's highest-rated movies.
        """
        # Get movies this user rated highly (≥ 4)
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        if user_ratings.empty:
            return pd.DataFrame()

        liked = user_ratings[user_ratings['rating'] >= 4].sort_values(
            'rating', ascending=False
        )
        if liked.empty:
            liked = user_ratings.sort_values('rating', ascending=False)

        rated_movie_ids = set(user_ratings['movie_id'].tolist())

        # Accumulate similarity scores across liked movies
        score_accumulator = {}

        for _, row in liked.head(5).iterrows():
            sims = self.get_similar_movies(row['movie_id'], top_n=50)
            for _, sim_row in sims.iterrows():
                mid = sim_row['movie_id']
                if mid in rated_movie_ids:
                    continue
                if mid not in score_accumulator:
                    score_accumulator[mid] = 0.0
                score_accumulator[mid] += sim_row['cb_similarity'] * row['rating']

        if not score_accumulator:
            return pd.DataFrame()

        # Sort and return top_n
        sorted_movies = sorted(score_accumulator.items(),
                               key=lambda x: x[1], reverse=True)[:top_n]
        movie_ids, scores = zip(*sorted_movies)

        recs = self.movies_df[self.movies_df['movie_id'].isin(movie_ids)][
            ['movie_id', 'title', 'genres']
        ].copy()

        score_map = dict(zip(movie_ids, scores))
        recs['cb_score'] = recs['movie_id'].map(score_map)

        # Normalize scores to [0, 1]
        max_s = recs['cb_score'].max()
        if max_s > 0:
            recs['cb_score_norm'] = recs['cb_score'] / max_s
        else:
            recs['cb_score_norm'] = 0.0

        return recs.sort_values('cb_score_norm', ascending=False).reset_index(drop=True)

    # ─────────────────────────────────────────────
    # Save / Load Model
    # ─────────────────────────────────────────────

    def save(self, path="models/cb_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[ContentBased] Model saved → {path}")

    @staticmethod
    def load(path="models/cb_model.pkl"):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"[ContentBased] Model loaded ← {path}")
        return model


# ─────────────────────────────────────────────
# Main – Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    movies  = pd.read_csv("data/processed/movies_clean.csv")
    ratings = pd.read_csv("data/processed/ratings_clean.csv")

    cb = ContentBasedFilter()
    cb.fit(movies)

    # Demo: find movies similar to movie_id=1
    print("\n[Demo] Movies similar to Movie 1:")
    print(cb.get_similar_movies(movie_id=1, top_n=5).to_string(index=False))

    # Demo: recommend for user 1
    print("\n[Demo] CB recommendations for User 1:")
    recs = cb.recommend_for_user(user_id=1, ratings_df=ratings, top_n=5)
    print(recs[['title', 'genres', 'cb_score_norm']].to_string(index=False))

    cb.save()
    print("\n✅  Content-Based Filtering complete.")
