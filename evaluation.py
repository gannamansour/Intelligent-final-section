"""
======================================================
 Hybrid Movie Recommendation System
 Script 5: Evaluation Metrics
         (RMSE, MAE, Precision@K, Recall@K, F1@K)
======================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from content_based import ContentBasedFilter
from collaborative_filtering import CollaborativeFilter
from hybrid_recommender import HybridRecommender


# ─────────────────────────────────────────────
# Rating Prediction Metrics
# ─────────────────────────────────────────────

def compute_rmse(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

def compute_mae(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


# ─────────────────────────────────────────────
# Ranking Metrics
# ─────────────────────────────────────────────

def precision_at_k(recommended, relevant, k):
    """Fraction of top-k recommended that are relevant."""
    rec_k = recommended[:k]
    hits  = len(set(rec_k) & set(relevant))
    return hits / k if k > 0 else 0.0

def recall_at_k(recommended, relevant, k):
    """Fraction of relevant items found in top-k."""
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits  = len(set(rec_k) & set(relevant))
    return hits / len(relevant)

def f1_at_k(p, r):
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ─────────────────────────────────────────────
# Evaluate Rating Prediction (RMSE / MAE)
# ─────────────────────────────────────────────

def evaluate_rating_prediction(model, test_df, model_name="Model"):
    """
    Evaluate RMSE and MAE for a CF or Hybrid model on the test set.
    """
    print(f"\n[Evaluating {model_name}] Rating Prediction …")
    preds = []
    for _, row in test_df.iterrows():
        if hasattr(model, 'predict_rating'):
            p = model.predict_rating(int(row['user_id']), int(row['movie_id']))
        else:
            p = model.predict(int(row['user_id']), int(row['movie_id']))
        preds.append(p)

    preds = np.array(preds, dtype=float)
    y     = test_df['rating'].values

    rmse = compute_rmse(y, preds)
    mae  = compute_mae(y, preds)

    valid = ~np.isnan(preds)
    print(f"  Evaluated on : {valid.sum():,} / {len(test_df):,} pairs")
    print(f"  RMSE         : {rmse:.4f}")
    print(f"  MAE          : {mae:.4f}")
    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'n_eval': int(valid.sum())}


# ─────────────────────────────────────────────
# Evaluate Ranking (Precision, Recall, F1)
# ─────────────────────────────────────────────

def evaluate_ranking(recommender, ratings_df, movies_df,
                     model_name="Model", k=10,
                     n_users=100, relevance_threshold=4.0):
    """
    Evaluate Precision@K, Recall@K, and F1@K.
    A movie is 'relevant' if the user gave it a rating ≥ relevance_threshold.
    """
    print(f"\n[Evaluating {model_name}] Ranking @ K={k} …")

    # Use a random sample of users that have enough data
    active_users = (
        ratings_df.groupby('user_id').size()
        .reset_index(name='n')
        .query('n >= 20')['user_id']
        .tolist()
    )
    np.random.seed(42)
    sampled = np.random.choice(active_users,
                               size=min(n_users, len(active_users)),
                               replace=False)

    precisions, recalls, f1s = [], [], []

    for uid in sampled:
        user_ratings = ratings_df[ratings_df['user_id'] == uid]

        # 80/20 split per user for hold-out evaluation
        shuffled = user_ratings.sample(frac=1, random_state=42)
        split    = int(len(shuffled) * 0.8)
        train_u  = shuffled.iloc[:split]
        test_u   = shuffled.iloc[split:]

        # Relevant items = those the user liked in test set
        relevant = test_u[test_u['rating'] >= relevance_threshold]['movie_id'].tolist()
        if not relevant:
            continue

        # Get recommendations (using train_u as "known" ratings)
        try:
            if isinstance(recommender, ContentBasedFilter):
                recs = recommender.recommend_for_user(uid, train_u, top_n=k)
                if recs.empty:
                    continue
                rec_ids = recs['movie_id'].tolist()
            else:
                recs = recommender.recommend(uid, train_u, top_n=k)
                if recs.empty:
                    continue
                rec_ids = recs['movie_id'].tolist()
        except Exception:
            continue

        p  = precision_at_k(rec_ids, relevant, k)
        r  = recall_at_k(rec_ids, relevant, k)
        f1 = f1_at_k(p, r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    mean_p  = float(np.mean(precisions)) if precisions else 0.0
    mean_r  = float(np.mean(recalls))    if recalls    else 0.0
    mean_f1 = float(np.mean(f1s))        if f1s        else 0.0

    print(f"  Users evaluated  : {len(precisions)}")
    print(f"  Precision@{k}     : {mean_p:.4f}")
    print(f"  Recall@{k}        : {mean_r:.4f}")
    print(f"  F1@{k}            : {mean_f1:.4f}")

    return {
        'model': model_name, f'precision@{k}': mean_p,
        f'recall@{k}': mean_r, f'f1@{k}': mean_f1,
        'n_users_eval': len(precisions)
    }


# ─────────────────────────────────────────────
# Full Evaluation Pipeline
# ─────────────────────────────────────────────

def full_evaluation():
    movies  = pd.read_csv("data/processed/movies_clean.csv")
    ratings = pd.read_csv("data/processed/ratings_clean.csv")
    train   = pd.read_csv("data/processed/train.csv")
    test    = pd.read_csv("data/processed/test.csv")

    print("=" * 55)
    print("  FULL EVALUATION PIPELINE")
    print("=" * 55)

    # ── Train models
    print("\n>> Training models …")
    cb = ContentBasedFilter()
    cb.fit(movies)

    cf = CollaborativeFilter(n_factors=50)
    cf.fit(train)

    hybrid_w = HybridRecommender(cb_weight=0.4, cf_weight=0.6, strategy='weighted')
    hybrid_w.fit(movies, train)

    hybrid_s = HybridRecommender(cb_weight=0.4, cf_weight=0.6, strategy='stacking')
    hybrid_s.fit(movies, train)

    # ── Rating prediction evaluation (CF + Hybrid only)
    rating_results = []
    rating_results.append(evaluate_rating_prediction(cf,      test, "CF (SVD)"))
    rating_results.append(evaluate_rating_prediction(hybrid_w, test, "Hybrid Weighted"))
    rating_results.append(evaluate_rating_prediction(hybrid_s, test, "Hybrid Stacking"))

    # ── Ranking evaluation (all models)
    rank_results = []
    rank_results.append(evaluate_ranking(cb,       ratings, movies, "Content-Based", k=10))
    rank_results.append(evaluate_ranking(hybrid_w, ratings, movies, "Hybrid Weighted", k=10))
    rank_results.append(evaluate_ranking(hybrid_s, ratings, movies, "Hybrid Stacking", k=10))

    # ── Summary Tables
    print("\n\n" + "=" * 55)
    print("  RATING PREDICTION SUMMARY")
    print("=" * 55)
    rating_df = pd.DataFrame(rating_results)
    print(rating_df.to_string(index=False))

    print("\n" + "=" * 55)
    print("  RANKING METRICS SUMMARY  (K=10)")
    print("=" * 55)
    rank_df = pd.DataFrame(rank_results)
    print(rank_df.to_string(index=False))

    # Save report
    os.makedirs("reports", exist_ok=True)
    rating_df.to_csv("reports/rating_prediction_metrics.csv", index=False)
    rank_df.to_csv("reports/ranking_metrics.csv", index=False)
    print("\n✅  Evaluation results saved to  reports/")

    return rating_df, rank_df


if __name__ == "__main__":
    full_evaluation()
