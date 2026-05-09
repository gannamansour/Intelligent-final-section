"""
======================================================
Hybrid Movie Recommendation System
Script 4: Hybrid Recommendation Engine
        (Weighted Average + Model Stacking options)
======================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from content_based import ContentBasedFilter
from collaborative_filtering import CollaborativeFilter


class HybridRecommender:
    """
    Hybrid Recommendation Engine combining:
    - Content-Based Filtering (CB)
    - Collaborative Filtering  (CF)

    Two strategies available:
    1. 'weighted'  – weighted average of normalized CB and CF scores
    2. 'stacking'  – CF score as base, CB as secondary for cold-start or boost
    """

    def __init__(self, cb_weight=0.4, cf_weight=0.6, strategy='weighted'):
        """
        Parameters
        ----------
        cb_weight : float  Weight for content-based scores  (0-1)
        cf_weight : float  Weight for collaborative scores   (0-1)
        strategy  : str    'weighted' or 'stacking'
        """
        assert strategy in ('weighted', 'stacking'), \
            "strategy must be 'weighted' or 'stacking'"
        assert abs(cb_weight + cf_weight - 1.0) < 1e-6, \
            "cb_weight + cf_weight must equal 1.0"

        self.cb_weight = cb_weight
        self.cf_weight = cf_weight
        self.strategy  = strategy
        self.cb_model  = None
        self.cf_model  = None
        self.movies_df = None

    # ─────────────────────────────────────────────
    # Train / Load Models
    # ─────────────────────────────────────────────

    def fit(self, movies_df, train_df):
        """Fit both sub-models."""
        self.movies_df = movies_df.copy()

        print("=" * 55)
        print("  TRAINING HYBRID RECOMMENDER")
        print("=" * 55)

        # Content-Based
        print("\n[1/2] Content-Based Filtering …")
        self.cb_model = ContentBasedFilter()
        self.cb_model.fit(movies_df)

        # Collaborative Filtering
        print("\n[2/2] Collaborative Filtering …")
        self.cf_model = CollaborativeFilter(n_factors=50)
        self.cf_model.fit(train_df)

        print("\n✅  Both models trained successfully.")

    def load_models(self, cb_path="models/cb_model.pkl",
                    cf_path="models/cf_model.pkl",
                    movies_df=None):
        """Load pre-trained models from disk."""
        self.cb_model  = ContentBasedFilter.load(cb_path)
        self.cf_model  = CollaborativeFilter.load(cf_path)
        self.movies_df = movies_df

    # ─────────────────────────────────────────────
    # Recommend
    # ─────────────────────────────────────────────

    def recommend(self, user_id, ratings_df, top_n=10):
        """
        Generate hybrid recommendations for a user.

        Returns a DataFrame with columns:
        movie_id, title, genres, cb_score_norm, cf_score_norm, hybrid_score
        """
        # ── Get CB recommendations
        cb_recs = self.cb_model.recommend_for_user(
            user_id, ratings_df, top_n=top_n * 5
        )

        # ── Get CF recommendations
        cf_recs = self.cf_model.recommend_for_user(
            user_id, ratings_df, self.movies_df, top_n=top_n * 5
        )

        # ── Cold-start fallback: if CF has no recs, return CB only
        if cf_recs.empty and not cb_recs.empty:
            cb_recs = cb_recs.head(top_n).copy()
            cb_recs['hybrid_score'] = cb_recs['cb_score_norm']
            return cb_recs

        # ── Cold-start fallback: if CB has no recs, return CF only
        if cb_recs.empty and not cf_recs.empty:
            cf_recs = cf_recs.head(top_n).copy()
            cf_recs['hybrid_score'] = cf_recs['cf_score_norm']
            return cf_recs

        if cb_recs.empty and cf_recs.empty:
            return pd.DataFrame()

        # ── Merge on movie_id
        merged = pd.merge(
            cb_recs[['movie_id', 'title', 'genres', 'cb_score_norm']],
            cf_recs[['movie_id', 'cf_score_norm', 'cf_pred_rating']],
            on='movie_id',
            how='outer'
        )

        # Fill NaN with 0 (movie present in only one model)
        merged['cb_score_norm'] = merged['cb_score_norm'].fillna(0.0)
        merged['cf_score_norm'] = merged['cf_score_norm'].fillna(0.0)

        # Fill missing titles/genres from movies_df
        movie_info = self.movies_df.set_index('movie_id')[['title', 'genres']]
        for idx, row in merged[merged['title'].isna()].iterrows():
            if row['movie_id'] in movie_info.index:
                merged.at[idx, 'title']  = movie_info.at[row['movie_id'], 'title']
                merged.at[idx, 'genres'] = movie_info.at[row['movie_id'], 'genres']

        # ── Compute Hybrid Score
        if self.strategy == 'weighted':
            merged['hybrid_score'] = (
                self.cb_weight * merged['cb_score_norm'] +
                self.cf_weight * merged['cf_score_norm']
            )

        elif self.strategy == 'stacking':
            # CF is base; CB acts as a multiplier / booster
            merged['hybrid_score'] = merged['cf_score_norm'] * (
                1 + self.cb_weight * merged['cb_score_norm']
            )
            # Normalize
            max_s = merged['hybrid_score'].max()
            if max_s > 0:
                merged['hybrid_score'] /= max_s

        # ── Sort and return top_n
        result = (
            merged
            .sort_values('hybrid_score', ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        return result

    # ─────────────────────────────────────────────
    # Predict single rating (CF-based)
    # ─────────────────────────────────────────────

    def predict_rating(self, user_id, movie_id):
        """Predict rating for a single (user, movie) pair."""
        return self.cf_model.predict(user_id, movie_id)

    # ─────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────

    def save(self, path="models/hybrid_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[Hybrid] Model saved → {path}")

    @staticmethod
    def load(path="models/hybrid_model.pkl"):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"[Hybrid] Model loaded ← {path}")
        return model


# ─────────────────────────────────────────────
# Main – Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    movies  = pd.read_csv("data/processed/movies_clean.csv")
    ratings = pd.read_csv("data/processed/ratings_clean.csv")
    train   = pd.read_csv("data/processed/train.csv")

    # Train
    hybrid = HybridRecommender(cb_weight=0.4, cf_weight=0.6, strategy='weighted')
    hybrid.fit(movies, train)

    # Recommend for User 1
    print("\n" + "=" * 55)
    print("  HYBRID RECOMMENDATIONS — User 1")
    print("=" * 55)
    recs = hybrid.recommend(user_id=1, ratings_df=ratings, top_n=10)
    cols = ['title', 'genres', 'cb_score_norm', 'cf_score_norm', 'hybrid_score']
    print(recs[cols].to_string(index=False))

    # Save
    hybrid.save()
    print("\n✅  Hybrid Recommendation Engine complete.")
