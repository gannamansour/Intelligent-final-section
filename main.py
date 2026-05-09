"""
======================================================
 Hybrid Movie Recommendation System
 Script 7: Main Pipeline Runner
         — runs all steps end-to-end
======================================================
Usage:  python main.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np


def print_banner(text):
    w = 60
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)


# ─────────────────────────────────────────────
# STEP 1: Data Preprocessing
# ─────────────────────────────────────────────
print_banner("STEP 1 — Data Preprocessing")
from data_preprocessing import (load_data, preprocess_movies,
                                 preprocess_ratings, explore_data,
                                 train_test_split)

movies_raw, ratings_raw = load_data()
movies  = preprocess_movies(movies_raw)
ratings = preprocess_ratings(ratings_raw)
explore_data(movies, ratings)
train_df, test_df = train_test_split(ratings)

os.makedirs("data/processed", exist_ok=True)
movies.to_csv("data/processed/movies_clean.csv",   index=False)
ratings.to_csv("data/processed/ratings_clean.csv", index=False)
train_df.to_csv("data/processed/train.csv",        index=False)
test_df.to_csv("data/processed/test.csv",          index=False)
print("✅  Data saved to data/processed/")


# ─────────────────────────────────────────────
# STEP 2: Content-Based Filtering
# ─────────────────────────────────────────────
print_banner("STEP 2 — Content-Based Filtering")
from content_based import ContentBasedFilter

cb = ContentBasedFilter()
cb.fit(movies)
cb.save("models/cb_model.pkl")

print("\n[Demo] Top 5 movies similar to Movie ID=1:")
sims = cb.get_similar_movies(movie_id=1, top_n=5)
print(sims[['title', 'genres', 'cb_similarity']].to_string(index=False))

print("\n[Demo] CB recommendations for User 1:")
cb_recs = cb.recommend_for_user(user_id=1, ratings_df=ratings, top_n=5)
if not cb_recs.empty:
    print(cb_recs[['title', 'genres', 'cb_score_norm']].to_string(index=False))
print("✅  Content-Based model done.")


# ─────────────────────────────────────────────
# STEP 3: Collaborative Filtering
# ─────────────────────────────────────────────
print_banner("STEP 3 — Collaborative Filtering (SVD)")
from collaborative_filtering import CollaborativeFilter

cf = CollaborativeFilter(n_factors=50)
cf.fit(train_df)
cf.save("models/cf_model.pkl")

p = cf.predict(user_id=1, movie_id=100)
print(f"\n[Demo] Predicted rating (User 1 → Movie 100): {p:.2f}" if p else
      "[Demo] (user,movie) pair not in training set")

print("\n[Demo] CF recommendations for User 1:")
cf_recs = cf.recommend_for_user(user_id=1, ratings_df=ratings,
                                movies_df=movies, top_n=5)
if not cf_recs.empty:
    print(cf_recs[['title', 'cf_pred_rating', 'cf_score_norm']].to_string(index=False))
print("✅  Collaborative Filtering model done.")


# ─────────────────────────────────────────────
# STEP 4: Hybrid Recommendation Engine
# ─────────────────────────────────────────────
print_banner("STEP 4 — Hybrid Recommendation Engine")
from hybrid_recommender import HybridRecommender

hybrid = HybridRecommender(cb_weight=0.4, cf_weight=0.6, strategy='weighted')
hybrid.fit(movies, train_df)
hybrid.save("models/hybrid_model.pkl")

print("\n[Demo] Hybrid recommendations for User 1:")
h_recs = hybrid.recommend(user_id=1, ratings_df=ratings, top_n=10)
if not h_recs.empty:
    cols = ['title', 'genres', 'cb_score_norm', 'cf_score_norm', 'hybrid_score']
    print(h_recs[cols].to_string(index=False))
print("✅  Hybrid model done.")


# ─────────────────────────────────────────────
# STEP 5: Evaluation
# ─────────────────────────────────────────────
print_banner("STEP 5 — Evaluation")
from evaluation import (evaluate_rating_prediction, evaluate_ranking,
                        full_evaluation)

rating_df, rank_df = full_evaluation()

print("\n" + "=" * 60)
print("  FINAL EVALUATION SUMMARY")
print("=" * 60)
print("\n[Rating Prediction]")
print(rating_df.to_string(index=False))
print("\n[Ranking Metrics @K=10]")
print(rank_df.to_string(index=False))


# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
print_banner("ALL STEPS COMPLETE ✅")
print("""
  Files created:
    data/processed/    — cleaned CSV files
    models/            — trained model .pkl files
    reports/           — evaluation metrics CSV files

  To launch the Streamlit UI:
    streamlit run app.py
""")
