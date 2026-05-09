"""
======================================================
 Hybrid Movie Recommendation System
 Script 1: Data Ingestion & Preprocessing
======================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────

def load_data(data_dir="data"):
    """Load movies and ratings from CSV files."""
    movies_path  = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")

    movies  = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    
    # Rename columns to snake_case for consistency
    movies.rename(columns={'movieId': 'movie_id'}, inplace=True)
    ratings.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'}, inplace=True)

    print("=" * 55)
    print("  DATA LOADING SUMMARY")
    print("=" * 55)
    print(f"  Movies  : {len(movies):,}")
    print(f"  Ratings : {len(ratings):,}")
    print(f"  Users   : {ratings['user_id'].nunique():,}")
    print("=" * 55)
    return movies, ratings


# ─────────────────────────────────────────────
# 2. Handle Missing Values & Data Consistency
# ─────────────────────────────────────────────

def preprocess_movies(movies_df):
    """Clean and validate the movies DataFrame."""
    df = movies_df.copy()

    # Fill missing titles
    missing_title = df['title'].isna().sum()
    df['title'] = df['title'].fillna(f"Unknown Movie")

    # Fill missing genres
    missing_genres = df['genres'].isna().sum()
    df['genres'] = df['genres'].fillna("Unknown")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['movie_id'])
    after  = len(df)

    # Ensure correct types
    df['movie_id'] = df['movie_id'].astype(int)

    print("\n[Movies Preprocessing]")
    print(f"  Missing titles  fixed : {missing_title}")
    print(f"  Missing genres  fixed : {missing_genres}")
    print(f"  Duplicates removed    : {before - after}")
    print(f"  Final movies count    : {len(df):,}")

    return df


def preprocess_ratings(ratings_df):
    """Clean and validate the ratings DataFrame."""
    df = ratings_df.copy()

    # Drop rows with NaN in critical columns
    before = len(df)
    df = df.dropna(subset=['user_id', 'movie_id', 'rating'])
    after  = len(df)
    print(f"\n[Ratings Preprocessing]")
    print(f"  Rows dropped (NaN)  : {before - after}")

    # Enforce rating range [1, 5]
    out_of_range = ((df['rating'] < 1) | (df['rating'] > 5)).sum()
    df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
    print(f"  Out-of-range ratings removed : {out_of_range}")

    # Remove duplicates (keep last)
    before = len(df)
    df = df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
    print(f"  Duplicate (user,movie) pairs removed : {before - len(df)}")

    # Ensure correct types
    df['user_id']  = df['user_id'].astype(int)
    df['movie_id'] = df['movie_id'].astype(int)
    df['rating']   = df['rating'].astype(float)

    print(f"  Final ratings count  : {len(df):,}")
    return df


# ─────────────────────────────────────────────
# 3. Exploratory Analysis
# ─────────────────────────────────────────────

def explore_data(movies_df, ratings_df):
    """Print basic EDA statistics."""
    print("\n" + "=" * 55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    # Rating distribution
    print("\n[Rating Distribution]")
    dist = ratings_df['rating'].value_counts().sort_index()
    for rating, count in dist.items():
        bar = "█" * (count // 2000)
        print(f"  {int(rating)}★  {count:6,}  {bar}")

    # Rating stats
    print(f"\n  Mean   : {ratings_df['rating'].mean():.3f}")
    print(f"  Median : {ratings_df['rating'].median():.1f}")
    print(f"  Std    : {ratings_df['rating'].std():.3f}")

    # Sparsity
    n_users  = ratings_df['user_id'].nunique()
    n_movies = ratings_df['movie_id'].nunique()
    sparsity = 1 - (len(ratings_df) / (n_users * n_movies))
    print(f"\n  Matrix Sparsity : {sparsity:.4%}")
    print(f"  Users           : {n_users:,}")
    print(f"  Movies rated    : {n_movies:,}")

    # Top genres
    genre_counts = {}
    for g_str in movies_df['genres'].dropna():
        for g in g_str.split('|'):
            genre_counts[g] = genre_counts.get(g, 0) + 1
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    print("\n[Top 10 Genres]")
    for genre, cnt in sorted_genres[:10]:
        bar = "█" * (cnt // 30)
        print(f"  {genre:<15}  {cnt:4}  {bar}")


# ─────────────────────────────────────────────
# 4. Train / Test Split
# ─────────────────────────────────────────────

def train_test_split(ratings_df, test_size=0.2, random_state=42):
    """Split ratings into 80/20 train-test sets."""
    from sklearn.model_selection import train_test_split as sk_split

    train, test = sk_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state
    )
    print(f"\n[Train/Test Split]")
    print(f"  Train : {len(train):,} ({(1-test_size)*100:.0f}%)")
    print(f"  Test  : {len(test):,}  ({test_size*100:.0f}%)")
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    movies_raw, ratings_raw = load_data()

    movies  = preprocess_movies(movies_raw)
    ratings = preprocess_ratings(ratings_raw)

    explore_data(movies, ratings)

    train_df, test_df = train_test_split(ratings)

    # Save preprocessed data
    os.makedirs("data/processed", exist_ok=True)
    movies.to_csv("data/processed/movies_clean.csv",   index=False)
    ratings.to_csv("data/processed/ratings_clean.csv", index=False)
    train_df.to_csv("data/processed/train.csv",        index=False)
    test_df.to_csv("data/processed/test.csv",          index=False)

    print("\n✅  Preprocessed data saved to  data/processed/")
