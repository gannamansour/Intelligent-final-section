"""
======================================================
 Hybrid Movie Recommendation System
 Script 3: Collaborative Filtering
         (Matrix Factorization via SVD)
======================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class CollaborativeFilter:
    """
    Collaborative Filtering using truncated SVD (matrix factorization).
    Decomposes the user-item rating matrix:  R ≈ U · S · Vt
    then uses the reconstructed matrix to predict unseen ratings.
    """

    def __init__(self, n_factors=50):
        """
        Parameters
        ----------
        n_factors : int
            Number of latent factors (singular values) to keep.
        """
        self.n_factors   = n_factors
        self.U           = None   # user factor matrix
        self.sigma       = None   # singular values
        self.Vt          = None   # item factor matrix
        self.predicted   = None   # full reconstructed rating matrix
        self.user_means  = None   # mean rating per user (for normalization)
        self.user_ids    = None   # ordered list of user_ids
        self.movie_ids   = None   # ordered list of movie_ids
        self.user_idx    = {}     # user_id → row index
        self.movie_idx   = {}     # movie_id → col index

    # Fit

    def fit(self, train_df):
        """
        Build the user-item matrix and perform truncated SVD.
        """
        print("[CollabFilter] Building user-item matrix …")

        # Create pivot table: rows=users, cols=movies, values=rating
        pivot = train_df.pivot_table(
            index='user_id', columns='movie_id', values='rating'
        )

        self.user_ids  = list(pivot.index)
        self.movie_ids = list(pivot.columns)
        self.user_idx  = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_idx = {m: i for i, m in enumerate(self.movie_ids)}

        # Fill NaN with 0 (mean-subtracted, so 0 = mean)
        R = pivot.values.copy()  # shape: (n_users, n_movies)

        # Subtract user mean (normalize)
        self.user_means = np.nanmean(R, axis=1)
        R_norm = R.copy()
        for i, mean in enumerate(self.user_means):
            mask = ~np.isnan(R[i])
            R_norm[i, mask] -= mean
        R_norm = np.nan_to_num(R_norm)  # replace remaining NaN with 0

        print(f"  Matrix shape : {R_norm.shape}")
        print(f"  Sparsity     : {(R_norm == 0).mean():.2%}")

        # Truncated SVD
        k = min(self.n_factors, min(R_norm.shape) - 1)
        print(f"  SVD factors  : {k}")
        self.U, self.sigma, self.Vt = svds(
            csr_matrix(R_norm).astype(float), k=k
        )
        # svds returns singular values in ascending order → reverse
        self.U      = self.U[:, ::-1]
        self.sigma  = self.sigma[::-1]
        self.Vt     = self.Vt[::-1, :]

        # Reconstruct normalized matrix
        R_pred_norm = np.dot(np.dot(self.U, np.diag(self.sigma)), self.Vt)

        # Add back user means
        R_pred = R_pred_norm + self.user_means.reshape(-1, 1)

        # Clip to [1, 5]
        self.predicted = np.clip(R_pred, 1, 5)

        print("[CollabFilter] SVD complete.")
        print(f"  Reconstructed matrix shape: {self.predicted.shape}")

    # ─────────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────────

    def predict_rating(self, user_id, movie_id):
        """Alias for predict() — used by evaluation pipeline."""
        return self.predict(user_id, movie_id)

    def predict(self, user_id, movie_id):
        """Predict rating for (user_id, movie_id). Returns float."""
        if user_id not in self.user_idx or movie_id not in self.movie_idx:
            return None
        u = self.user_idx[user_id]
        m = self.movie_idx[movie_id]
        return float(self.predicted[u, m])

    def predict_batch(self, pairs_df):
        """
        Predict ratings for a DataFrame with columns [user_id, movie_id].
        Returns a Series of predicted ratings.
        """
        preds = []
        for _, row in pairs_df.iterrows():
            p = self.predict(row['user_id'], row['movie_id'])
            preds.append(p if p is not None else np.nan)
        return pd.Series(preds, index=pairs_df.index)

    # ─────────────────────────────────────────────
    # Recommend for a User
    # ─────────────────────────────────────────────

    def recommend_for_user(self, user_id, ratings_df, movies_df, top_n=10):
        """
        Return top_n unseen movies for user_id, sorted by predicted rating.
        """
        if user_id not in self.user_idx:
            return pd.DataFrame()

        u = self.user_idx[user_id]
        pred_row = self.predicted[u]   # predicted ratings for all trained movies

        # Movies the user has already rated
        rated_ids = set(
            ratings_df[ratings_df['user_id'] == user_id]['movie_id'].tolist()
        )

        # Build dataframe of predictions
        recs = []
        for movie_id, col_idx in self.movie_idx.items():
            if movie_id in rated_ids:
                continue
            recs.append({'movie_id': movie_id, 'cf_pred_rating': pred_row[col_idx]})

        if not recs:
            return pd.DataFrame()

        recs_df = pd.DataFrame(recs).sort_values('cf_pred_rating', ascending=False)
        recs_df = recs_df.head(top_n)

        # Normalize to [0, 1]
        min_r, max_r = recs_df['cf_pred_rating'].min(), recs_df['cf_pred_rating'].max()
        if max_r > min_r:
            recs_df['cf_score_norm'] = (recs_df['cf_pred_rating'] - min_r) / (max_r - min_r)
        else:
            recs_df['cf_score_norm'] = 1.0

        # Merge with movie info
        recs_df = recs_df.merge(
            movies_df[['movie_id', 'title', 'genres']], on='movie_id', how='left'
        )
        return recs_df.reset_index(drop=True)

    # ─────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────

    def save(self, path="models/cf_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[CollabFilter] Model saved → {path}")

    @staticmethod
    def load(path="models/cf_model.pkl"):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"[CollabFilter] Model loaded ← {path}")
        return model


# ─────────────────────────────────────────────
# Main – Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train   = pd.read_csv("data/processed/train.csv")
    ratings = pd.read_csv("data/processed/ratings_clean.csv")
    movies  = pd.read_csv("data/processed/movies_clean.csv")

    cf = CollaborativeFilter(n_factors=50)
    cf.fit(train)

    # Demo: predict rating for (user=1, movie=10)
    p = cf.predict(user_id=1, movie_id=10)
    print(f"\n[Demo] Predicted rating for User 1 → Movie 10 : {p:.2f}")

    # Demo: top-5 recommendations for user 1
    print("\n[Demo] CF recommendations for User 1:")
    recs = cf.recommend_for_user(user_id=1, ratings_df=ratings,
                                 movies_df=movies, top_n=5)
    print(recs[['title', 'genres', 'cf_pred_rating', 'cf_score_norm']].to_string(index=False))

    cf.save()
    print("\n✅  Collaborative Filtering complete.")
