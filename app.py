"""
======================================================
 Hybrid Movie Recommendation System
 Script 6: Streamlit UI — Cinematic Edition
======================================================
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from content_based import ContentBasedFilter
from collaborative_filtering import CollaborativeFilter
from hybrid_recommender import HybridRecommender
from evaluation import evaluate_rating_prediction, evaluate_ranking


st.set_page_config(
    page_title="CineMatch — Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0rem !important; padding-bottom: 2rem !important; max-width: 1280px !important; }

.stApp {
    background: #0a0a0a !important;
    background-image: radial-gradient(ellipse 80% 50% at 50% -10%, #c8960015 0%, transparent 60%) !important;
}

[data-testid="stSidebar"] { background: #0d0d0d !important; border-right: 1px solid #c8960020 !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem !important; }

/* Hide sidebar collapse button to prevent collapsing */
[data-testid="stSidebar"] button[kind="header"] { display: none !important; visibility: hidden !important; }
[data-testid="collapsedControl"] { display: none !important; visibility: hidden !important; }
section[data-testid="stSidebar"] > button { display: none !important; visibility: hidden !important; }
[data-testid="stSidebar"] > button:first-child { display: none !important; visibility: hidden !important; }
button[aria-label="Close sidebar"] { display: none !important; visibility: hidden !important; }
button[aria-label="Collapse sidebar"] { display: none !important; visibility: hidden !important; }
[data-testid="baseButton-header"] { display: none !important; visibility: hidden !important; }
[data-testid="stSidebar"] [data-testid="baseButton-header"] { display: none !important; visibility: hidden !important; }

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label { color: #555 !important; font-family: 'DM Sans', sans-serif !important; font-size: 12px !important; }

[data-testid="stSidebar"] .stRadio > div { gap: 4px !important; }
[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important; border: none !important;
    border-left: 2px solid transparent !important; border-radius: 0 !important;
    padding: 10px 16px !important; color: #444 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; font-weight: 500 !important;
    transition: all 0.2s !important; cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    border-left-color: #c89600 !important; color: #c89600 !important; background: #c8960008 !important;
}

.stSlider > div > div > div > div { background: #c89600 !important; }
.stSlider > div > div > div > div > div { background: #f0c040 !important; border-color: #f0c040 !important; }

.stButton > button {
    background: transparent !important; color: #c89600 !important;
    border: 1px solid #c89600 !important; border-radius: 3px !important;
    padding: 10px 28px !important; font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important; font-size: 12px !important; letter-spacing: 2px !important;
    text-transform: uppercase !important; transition: all 0.25s !important; width: 100% !important;
}
.stButton > button:hover { background: #c89600 !important; color: #0a0a0a !important; box-shadow: 0 0 30px #c8960040 !important; }

.stNumberInput input, .stTextInput input {
    background: #111 !important; border: 1px solid #222 !important; border-radius: 3px !important;
    color: #ddd !important; font-family: 'DM Mono', monospace !important; font-size: 15px !important;
}
.stNumberInput input:focus, .stTextInput input:focus { border-color: #c89600 !important; box-shadow: 0 0 0 1px #c8960030 !important; }

.stSelectbox > div > div { background: #111 !important; border: 1px solid #222 !important; border-radius: 3px !important; color: #ddd !important; }

.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1e1e1e !important; border-radius: 0 !important; padding: 0 !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #444 !important; border-radius: 0 !important; border-bottom: 2px solid transparent !important; font-family: 'DM Mono', monospace !important; font-size: 10px !important; letter-spacing: 2px !important; text-transform: uppercase !important; padding: 12px 20px !important; margin-bottom: -1px !important; transition: all 0.2s !important; }
.stTabs [aria-selected="true"] { background: transparent !important; color: #c89600 !important; border-bottom: 2px solid #c89600 !important; }

.streamlit-expanderHeader { background: #111 !important; border: 1px solid #1e1e1e !important; border-radius: 3px !important; color: #666 !important; font-family: 'DM Mono', monospace !important; font-size: 10px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
.streamlit-expanderContent { background: #0d0d0d !important; border: 1px solid #1a1a1a !important; border-top: none !important; }

[data-testid="stMetricValue"] { color: #f0c040 !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 2.2rem !important; letter-spacing: 1px !important; }
[data-testid="stMetricLabel"] { color: #444 !important; font-family: 'DM Mono', monospace !important; font-size: 10px !important; letter-spacing: 2px !important; text-transform: uppercase !important; }
[data-testid="metric-container"] { background: #111 !important; border: 1px solid #1e1e1e !important; border-top: 2px solid #c89600 !important; border-radius: 0 0 3px 3px !important; padding: 1.2rem 1rem !important; }

.stSuccess { background: #0a1a0f !important; border: 1px solid #22c55e25 !important; border-left: 3px solid #22c55e !important; border-radius: 0 3px 3px 0 !important; }
.stWarning { background: #1a1200 !important; border: 1px solid #c8960030 !important; border-left: 3px solid #c89600 !important; border-radius: 0 3px 3px 0 !important; }
.stInfo { background: #0a0f1a !important; border: 1px solid #3b82f625 !important; border-left: 3px solid #3b82f6 !important; border-radius: 0 3px 3px 0 !important; }

[data-testid="stVegaLiteChart"] { background: #111 !important; border-radius: 3px !important; border: 1px solid #1e1e1e !important; padding: 12px !important; }
.stSpinner > div { border-top-color: #c89600 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #c89600; }
hr { border-color: #1e1e1e !important; }

/* ── Custom components */
.hero-wrap { padding: 2.5rem 0 2rem; border-bottom: 1px solid #1a1a1a; margin-bottom: 2rem; }
.hero-eyebrow { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 4px; text-transform: uppercase; color: #c89600; margin-bottom: 10px; }
.hero-title { font-family: 'Bebas Neue', sans-serif; font-size: clamp(3.5rem, 8vw, 7rem); line-height: 0.95; letter-spacing: 3px; color: #f5f5f0; text-shadow: 0 0 60px #c8960018; }
.hero-title span { color: #c89600; }
.hero-sub { font-family: 'DM Sans', sans-serif; font-size: 13px; color: #444; margin-top: 10px; letter-spacing: 0.5px; }

.sidebar-logo { font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; letter-spacing: 4px; color: #c89600; padding: 0 1rem 1.5rem; border-bottom: 1px solid #1e1e1e; margin-bottom: 1.5rem; }
.sidebar-logo span { color: #333; font-size: 0.9rem; letter-spacing: 2px; display: block; font-family: 'DM Mono', monospace; margin-top: -4px; }

.section-label { font-family: 'DM Mono', monospace; font-size: 9px; font-weight: 500; letter-spacing: 3px; text-transform: uppercase; color: #c89600; margin-bottom: 14px; margin-top: 6px; display: flex; align-items: center; gap: 10px; }
.section-label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, #c8960030, transparent); }

.movie-card { background: #0e0e0e; border: 1px solid #1a1a1a; border-left: 3px solid #c89600; border-radius: 0 4px 4px 0; padding: 20px 24px; margin: 10px 0; position: relative; transition: border-color 0.2s, background 0.2s; }
.movie-card:hover { background: #111; border-left-color: #f0c040; }
.movie-num { font-family: 'Bebas Neue', sans-serif; font-size: 3rem; color: #1e1e1e; position: absolute; top: 10px; right: 20px; line-height: 1; letter-spacing: 2px; user-select: none; }
.movie-rank { font-family: 'DM Mono', monospace; font-size: 9.5px; font-weight: 500; letter-spacing: 2px; text-transform: uppercase; color: #c89600; margin-bottom: 6px; }
.movie-title { font-family: 'DM Sans', sans-serif; font-size: 19px; font-weight: 600; color: #efefef; margin-bottom: 10px; line-height: 1.3; padding-right: 50px; }
.genre-tag { display: inline-block; background: transparent; color: #555; border: 1px solid #252525; border-radius: 2px; padding: 2px 8px; font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; margin: 2px 3px 2px 0; }
.score-row { display: flex; gap: 20px; margin-top: 16px; padding-top: 14px; border-top: 1px solid #1a1a1a; }
.score-item { flex: 1; }
.score-label { font-family: 'DM Mono', monospace; font-size: 8.5px; letter-spacing: 2px; text-transform: uppercase; color: #333; margin-bottom: 4px; }
.score-value { font-family: 'Bebas Neue', sans-serif; font-size: 1.4rem; letter-spacing: 1px; color: #c89600; margin-bottom: 4px; line-height: 1; }
.score-track { background: #1a1a1a; border-radius: 1px; height: 2px; overflow: hidden; }
.score-fill { height: 2px; border-radius: 1px; }

.sstat { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; margin: 4px 0; border-bottom: 1px solid #161616; }
.sstat-l { font-family: 'DM Sans', sans-serif; font-size: 11px; color: #444; }
.sstat-r { font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500; color: #c89600; }

.stat-chip { display: inline-block; background: #111; border: 1px solid #1e1e1e; border-top: 1px solid #c8960030; border-radius: 3px; padding: 8px 14px; font-family: 'DM Sans', sans-serif; font-size: 12px; color: #666; margin: 3px 4px 3px 0; }
.stat-chip b { color: #c89600; font-weight: 600; }

.h-item { display: flex; justify-content: space-between; align-items: center; background: #0d0d0d; border: 1px solid #161616; border-radius: 3px; padding: 10px 14px; margin: 5px 0; }
.h-title { font-family: 'DM Sans', sans-serif; font-size: 13px; font-weight: 500; color: #bbb; }
.h-genre { font-family: 'DM Mono', monospace; font-size: 10px; color: #333; margin-top: 3px; letter-spacing: 1px; text-transform: uppercase; }
.h-star { font-size: 12px; letter-spacing: 2px; }

.nav-label { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 3px; text-transform: uppercase; color: #333; padding: 0 1rem; margin-bottom: 8px; }

/* ── Evaluation components */
.eval-card {
    background: #0e0e0e; border: 1px solid #1a1a1a;
    border-radius: 4px; padding: 24px 28px; margin: 10px 0;
    position: relative; overflow: hidden;
}
.eval-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #c89600, #f0c040, #c89600);
}
.eval-model-name {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem;
    letter-spacing: 2px; color: #f5f5f0; margin-bottom: 16px;
}
.eval-metric-row { display: flex; gap: 16px; flex-wrap: wrap; }
.eval-metric-box {
    flex: 1; min-width: 100px; background: #111;
    border: 1px solid #1e1e1e; border-radius: 3px; padding: 14px 16px;
    text-align: center;
}
.eval-metric-val {
    font-family: 'Bebas Neue', sans-serif; font-size: 2rem;
    letter-spacing: 1px; color: #f0c040; line-height: 1;
}
.eval-metric-lbl {
    font-family: 'DM Mono', monospace; font-size: 9px;
    letter-spacing: 2px; text-transform: uppercase; color: #444; margin-top: 5px;
}
.eval-badge {
    display: inline-block; font-family: 'DM Mono', monospace; font-size: 9px;
    letter-spacing: 2px; text-transform: uppercase; padding: 3px 10px;
    border-radius: 2px; margin-bottom: 12px;
}
.eval-badge-best { background: #c8960020; color: #c89600; border: 1px solid #c8960040; }
.eval-badge-good { background: #3b82f615; color: #3b82f6; border: 1px solid #3b82f630; }
.eval-badge-base { background: #ffffff08; color: #555; border: 1px solid #ffffff12; }

.insight-box {
    background: #0d0d0d; border: 1px solid #1a1a1a;
    border-left: 3px solid #c89600; border-radius: 0 4px 4px 0;
    padding: 16px 20px; margin: 8px 0;
}
.insight-title { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: #c89600; margin-bottom: 6px; }
.insight-text { font-family: 'DM Sans', sans-serif; font-size: 13px; color: #888; line-height: 1.6; }

.compare-bar-wrap { margin: 6px 0; }
.compare-bar-label { display: flex; justify-content: space-between; font-family: 'DM Mono', monospace; font-size: 10px; color: #555; letter-spacing: 1px; margin-bottom: 3px; }
.compare-bar-track { background: #1a1a1a; border-radius: 1px; height: 6px; overflow: hidden; }
.compare-bar-fill { height: 6px; border-radius: 1px; transition: width 0.6s ease; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load Data & Models
# ─────────────────────────────────────────────

@st.cache_resource
def load_all():
    movies  = pd.read_csv("data/processed/movies_clean.csv")
    ratings = pd.read_csv("data/processed/ratings_clean.csv")
    train   = pd.read_csv("data/processed/train.csv")
    hybrid  = HybridRecommender(cb_weight=0.4, cf_weight=0.6, strategy='weighted')
    hybrid.fit(movies, train)
    hybrid.movies_df = movies
    return movies, ratings, hybrid

@st.cache_resource
def load_eval_data():
    """Load precomputed evaluation CSVs if available."""
    rating_csv  = "reports/rating_prediction_metrics.csv"
    ranking_csv = "reports/ranking_metrics.csv"
    rating_df  = pd.read_csv(rating_csv)  if os.path.exists(rating_csv)  else None
    ranking_df = pd.read_csv(ranking_csv) if os.path.exists(ranking_csv) else None
    return rating_df, ranking_df


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def genre_tags(s):
    return "".join(
        f'<span class="genre-tag">{g.strip()}</span>'
        for g in str(s).split('|') if g.strip()
    )

def score_block(val, label, c1, c2):
    pct = int(float(val) * 100)
    return (
        f'<div class="score-item">'
        f'<div class="score-label">{label}</div>'
        f'<div class="score-value">{float(val):.3f}</div>'
        f'<div class="score-track"><div class="score-fill" style="width:{pct}%;background:linear-gradient(90deg,{c1},{c2})"></div></div>'
        f'</div>'
    )

def star_str(r):
    r = int(r)
    return "★" * r + "☆" * (5 - r)

def compare_bar(label, value, max_val, color="#c89600"):
    pct = (value / max_val * 100) if max_val > 0 else 0
    return (
        f'<div class="compare-bar-wrap">'
        f'<div class="compare-bar-label"><span>{label}</span><span>{value:.4f}</span></div>'
        f'<div class="compare-bar-track"><div class="compare-bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
        f'</div>'
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    with st.spinner("Loading models…"):
        movies, ratings, hybrid = load_all()

    # ── Hero ──────────────────────────────────
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">✦ AI-Powered · Hybrid Engine · Content + Collaborative</div>
        <div class="hero-title">Cine<span>Match</span></div>
        <div class="hero-sub">Hybrid Movie Recommendation System &nbsp;·&nbsp; Content-Based Filtering + Collaborative Filtering + SVD</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">CM<span>// CINEMATCH</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
        mode = st.radio("", [
            "▸  Rate Movies & Get Recommendations",
            "▸  User Recommendations",
            "▸  Similar Movies",
            "▸  Explore Dataset",
            "▸  Evaluation Report",
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="nav-label">Model Parameters</div>', unsafe_allow_html=True)
        top_n    = st.slider("Recommendations", 5, 20, 10)
        strategy = st.selectbox("Hybrid Strategy", ["weighted", "stacking"])
        cb_w     = st.slider("Content Weight", 0.0, 1.0, 0.4, 0.05)
        cf_w     = round(1.0 - cb_w, 2)
        hybrid.cb_weight = cb_w
        hybrid.cf_weight = cf_w
        hybrid.strategy  = strategy
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="nav-label">Dataset Stats</div>', unsafe_allow_html=True)
        for lbl, val in [
            ("Movies",    f"{len(movies):,}"),
            ("Ratings",   f"{len(ratings):,}"),
            ("Users",     f"{ratings['user_id'].nunique():,}"),
            ("CB Weight", f"{cb_w:.0%}"),
            ("CF Weight", f"{cf_w:.0%}"),
            ("Strategy",  strategy.title()),
        ]:
            st.markdown(
                f'<div class="sstat"><span class="sstat-l">{lbl}</span>'
                f'<span class="sstat-r">{val}</span></div>',
                unsafe_allow_html=True
            )

    # ══════════════════════════════════════════
    # MODE 0 — Rate Movies & Get Recommendations (New User Input)
    # ══════════════════════════════════════════
    if "Rate Movies" in mode:
        st.markdown('<div class="section-label">Rate Movies & Get Your Recommendations</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="padding:12px 20px;background:#0d0d0d;border:1px solid #1a1a1a;border-left:3px solid #c89600;border-radius:0 4px 4px 0;margin:10px 0;">
            <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#c89600;margin-bottom:6px;">
                How It Works
            </div>
            <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:#888;line-height:1.6;">
                Rate at least 5 movies below to get personalized recommendations. The more movies you rate, the better your recommendations will be!
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for user ratings
        if 'user_ratings' not in st.session_state:
            st.session_state.user_ratings = {}
        
        # Get popular movies for rating
        popular_movies = (
            ratings.groupby('movie_id')
            .agg({'rating': ['count', 'mean']})
            .reset_index()
        )
        popular_movies.columns = ['movie_id', 'rating_count', 'avg_rating']
        popular_movies = popular_movies[popular_movies['rating_count'] >= 50]
        popular_movies = popular_movies.merge(movies[['movie_id', 'title', 'genres']], on='movie_id')
        popular_movies = popular_movies.sort_values('rating_count', ascending=False).head(100)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Select & Rate Movies</div>', unsafe_allow_html=True)
        
        # Search/filter movies
        search_term = st.text_input("🔍 Search movies by title:", "")
        
        if search_term:
            filtered_movies = popular_movies[
                popular_movies['title'].str.contains(search_term, case=False, na=False)
            ]
        else:
            filtered_movies = popular_movies.head(30)
        
        # Display movies for rating
        st.markdown(f"<div style='color:#666;font-size:12px;margin:10px 0;'>Showing {len(filtered_movies)} movies</div>", unsafe_allow_html=True)
        
        # Create rating interface
        num_cols = 2
        cols = st.columns(num_cols)
        
        for idx, (_, movie) in enumerate(filtered_movies.iterrows()):
            col_idx = idx % num_cols
            with cols[col_idx]:
                movie_id = int(movie['movie_id'])
                
                # Movie card
                st.markdown(
                    f'<div style="background:#0e0e0e;border:1px solid #1a1a1a;border-radius:4px;padding:12px 16px;margin:8px 0;">'
                    f'<div style="font-family:\'DM Sans\',sans-serif;font-size:14px;font-weight:600;color:#efefef;margin-bottom:6px;">{movie["title"]}</div>'
                    f'<div style="font-size:11px;color:#555;margin-bottom:8px;">{str(movie["genres"]).replace("|", " · ")}</div>'
                    f'<div style="font-size:10px;color:#444;">Avg: {movie["avg_rating"]:.1f}★ · {int(movie["rating_count"])} ratings</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Rating selector
                current_rating = st.session_state.user_ratings.get(movie_id, 0)
                rating = st.select_slider(
                    f"Rate {movie['title'][:30]}...",
                    options=[0, 1, 2, 3, 4, 5],
                    value=current_rating,
                    format_func=lambda x: "Not Rated" if x == 0 else f"{x}★",
                    key=f"rating_{movie_id}",
                    label_visibility="collapsed"
                )
                
                if rating > 0:
                    st.session_state.user_ratings[movie_id] = rating
                elif movie_id in st.session_state.user_ratings and rating == 0:
                    del st.session_state.user_ratings[movie_id]
        
        # Show current ratings summary
        st.markdown("<br>", unsafe_allow_html=True)
        num_ratings = len(st.session_state.user_ratings)
        
        if num_ratings > 0:
            st.markdown(
                f'<div style="padding:16px 20px;background:#111;border:1px solid #1e1e1e;border-top:2px solid #c89600;border-radius:0 0 3px 3px;margin:10px 0;">'
                f'<div style="font-family:\'Bebas Neue\',sans-serif;font-size:2.2rem;letter-spacing:1px;color:#f0c040;">{num_ratings}</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#444;">Movies Rated</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Show rated movies
            with st.expander(f"▸ View Your {num_ratings} Ratings"):
                for movie_id, rating in sorted(st.session_state.user_ratings.items(), key=lambda x: x[1], reverse=True):
                    movie_info = movies[movies['movie_id'] == movie_id].iloc[0]
                    star_color = "#f0c040" if rating >= 4 else ("#888" if rating >= 3 else "#444")
                    st.markdown(
                        f'<div class="h-item">'
                        f'<div><div class="h-title">{movie_info["title"]}</div>'
                        f'<div class="h-genre">{str(movie_info["genres"]).replace("|", " · ")}</div></div>'
                        f'<div class="h-star" style="color:{star_color}">{star_str(rating)}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        
        # Get Recommendations button
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            get_recs_btn = st.button("✨ Get My Recommendations", disabled=(num_ratings < 5))
        
        if num_ratings < 5:
            st.warning(f"⚠️ Please rate at least 5 movies to get recommendations. You've rated {num_ratings} so far.")
        
        # Generate recommendations
        if get_recs_btn and num_ratings >= 5:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Your Personalized Recommendations</div>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing your taste and generating recommendations…"):
                # Create a temporary user ID (use max user_id + 1)
                temp_user_id = ratings['user_id'].max() + 1
                
                # Create a DataFrame with user's ratings
                user_ratings_df = pd.DataFrame([
                    {'user_id': temp_user_id, 'movie_id': mid, 'rating': float(r)}
                    for mid, r in st.session_state.user_ratings.items()
                ])
                
                # Combine with existing ratings for the recommendation engine
                combined_ratings = pd.concat([ratings, user_ratings_df], ignore_index=True)
                
                # Get recommendations
                try:
                    recs = hybrid.recommend(user_id=temp_user_id, ratings_df=combined_ratings, top_n=top_n)
                    
                    if recs.empty:
                        st.warning("Unable to generate recommendations. Please try rating more movies.")
                    else:
                        # Calculate user's genre preferences
                        user_rated_movies = movies[movies['movie_id'].isin(st.session_state.user_ratings.keys())]
                        genre_counts = {}
                        for genres_str in user_rated_movies['genres']:
                            for g in str(genres_str).split('|'):
                                genre_counts[g] = genre_counts.get(g, 0) + 1
                        top_genre = max(genre_counts, key=genre_counts.get) if genre_counts else "Various"
                        avg_rating = np.mean(list(st.session_state.user_ratings.values()))
                        
                        st.markdown(
                            f'<div style="padding:16px 0">'
                            f'<span class="stat-chip">🎬 <b>{num_ratings}</b> movies rated</span>'
                            f'<span class="stat-chip">⭐ avg <b>{avg_rating:.2f}</b></span>'
                            f'<span class="stat-chip">🎭 favorite genre <b>{top_genre}</b></span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="section-label">Top {len(recs)} Recommendations For You</div>',
                            unsafe_allow_html=True
                        )
                        
                        for i, row in recs.iterrows():
                            cb_s = float(row.get('cb_score_norm', 0) or 0)
                            cf_s = float(row.get('cf_score_norm', 0) or 0)
                            hy_s = float(row.get('hybrid_score', 0) or 0)
                            st.markdown(
                                f'<div class="movie-card">'
                                f'<div class="movie-num">{i+1:02d}</div>'
                                f'<div class="movie-rank">Rank #{i+1} · Match Score {hy_s:.3f}</div>'
                                f'<div class="movie-title">{row["title"]}</div>'
                                f'<div>{genre_tags(row.get("genres", ""))}</div>'
                                f'<div class="score-row">'
                                f'{score_block(cb_s, "Content Match", "#c89600", "#f0c040")}'
                                f'{score_block(cf_s, "User Similarity", "#888", "#bbb")}'
                                f'{score_block(hy_s, "Overall Match", "#c89600", "#f0c040")}'
                                f'</div></div>',
                                unsafe_allow_html=True
                            )
                        
                        st.success(f"✅ Generated {len(recs)} personalized recommendations based on your ratings!")
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
                    st.info("Try rating more popular movies or adjusting your ratings.")
        
        # Clear ratings button
        if num_ratings > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🗑️ Clear All Ratings"):
                    st.session_state.user_ratings = {}
                    st.rerun()

    # ══════════════════════════════════════════
    # MODE 1 — User Recommendations
    # ══════════════════════════════════════════
    elif "User" in mode:
        st.markdown('<div class="section-label">Personalized Recommendations</div>', unsafe_allow_html=True)

        # Automatically select a random user for demonstration
        available_users = sorted(ratings['user_id'].unique().tolist())
        
        # Use session state to maintain the same user across reruns
        if 'selected_user' not in st.session_state:
            st.session_state.selected_user = available_users[0]
        
        user_id = st.session_state.selected_user
        
        # Add a button to get recommendations for a different user
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🎲 Random User"):
                import random
                st.session_state.selected_user = random.choice(available_users)
                st.rerun()
        
        st.markdown(f'<div style="padding:8px 0;color:#888;font-size:13px;">Showing recommendations for <b style="color:#c89600;">User #{user_id}</b></div>', unsafe_allow_html=True)

        if user_id:
            ur = ratings[ratings['user_id'] == user_id]
            if not ur.empty:
                um = ur.merge(movies[['movie_id', 'genres','title']], on='movie_id')
                gc = {}
                for gs in um['genres']:
                    for g in str(gs).split('|'):
                        gc[g] = gc.get(g, 0) + 1
                top_g = max(gc, key=gc.get) if gc else "—"
                st.markdown(
                    f'<div style="padding:16px 0">'
                    f'<span class="stat-chip">🎬 <b>{len(ur):,}</b> rated</span>'
                    f'<span class="stat-chip">⭐ avg <b>{ur["rating"].mean():.2f}</b></span>'
                    f'<span class="stat-chip">🎭 top genre <b>{top_g}</b></span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with st.spinner("Computing recommendations…"):
                recs = hybrid.recommend(user_id=user_id, ratings_df=ratings, top_n=top_n)

            if recs.empty:
                st.warning("No recommendations found for this user.")
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander(f"▸ User #{user_id} — Top Rated Movies"):
                    top5 = (
                        ratings[ratings['user_id'] == user_id]
                        .sort_values('rating', ascending=False)
                        .head(5)
                        .merge(movies[['movie_id', 'title', 'genres']], on='movie_id')
                    )
                    for _, r in top5.iterrows():
                        star_color = "#f0c040" if r['rating'] >= 4 else ("#666" if r['rating'] >= 3 else "#333")
                        st.markdown(
                            f'<div class="h-item">'
                            f'<div><div class="h-title">{r["title"]}</div>'
                            f'<div class="h-genre">{str(r["genres"]).replace("|", " · ")}</div></div>'
                            f'<div class="h-star" style="color:{star_color}">{star_str(r["rating"])}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="section-label">Top {len(recs)} Picks — User #{user_id}</div>',
                    unsafe_allow_html=True
                )
                for i, row in recs.iterrows():
                    cb_s = float(row.get('cb_score_norm', 0) or 0)
                    cf_s = float(row.get('cf_score_norm', 0) or 0)
                    hy_s = float(row.get('hybrid_score',  0) or 0)
                    st.markdown(
                        f'<div class="movie-card">'
                        f'<div class="movie-num">{i+1:02d}</div>'
                        f'<div class="movie-rank">Rank #{i+1} · Hybrid Score {hy_s:.3f}</div>'
                        f'<div class="movie-title">{row["title"]}</div>'
                        f'<div>{genre_tags(row.get("genres", ""))}</div>'
                        f'<div class="score-row">'
                        f'{score_block(cb_s, "Content-Based", "#c89600", "#f0c040")}'
                        f'{score_block(cf_s, "Collaborative", "#888", "#bbb")}'
                        f'{score_block(hy_s, "Hybrid", "#c89600", "#f0c040")}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

    # ══════════════════════════════════════════
    # MODE 2 — Similar Movies
    # ══════════════════════════════════════════
    elif "Similar" in mode:
        st.markdown('<div class="section-label">Content-Based Similarity Search</div>', unsafe_allow_html=True)

        titles = sorted(movies['title'].tolist())
        sel    = st.selectbox("Select a movie:", titles)
        mrow   = movies[movies['title'] == sel].iloc[0]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<div style='margin:6px 0 0'>{genre_tags(mrow['genres'])}</div>", unsafe_allow_html=True)
        with col2:
            run = st.button("▸ Find Similar")

        if run:
            with st.spinner("Computing similarities…"):
                sims = hybrid.cb_model.get_similar_movies(int(mrow['movie_id']), top_n=top_n)

            if sims.empty:
                st.warning("No similar movies found.")
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="section-label">Top {len(sims)} Similar to "{sel}"</div>', unsafe_allow_html=True)
                for i, row in sims.iterrows():
                    sim = float(row['cb_similarity'])
                    st.markdown(
                        f'<div class="movie-card">'
                        f'<div class="movie-num">{i+1:02d}</div>'
                        f'<div class="movie-rank">Rank #{i+1} · Cosine Similarity {sim:.4f}</div>'
                        f'<div class="movie-title">{row["title"]}</div>'
                        f'<div>{genre_tags(row["genres"])}</div>'
                        f'<div class="score-row">'
                        f'{score_block(sim, "TF-IDF Cosine Similarity", "#c89600", "#f0c040")}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

    # ══════════════════════════════════════════
    # MODE 3 — Explore Dataset
    # ══════════════════════════════════════════
    elif "Explore" in mode:
        st.markdown('<div class="section-label">Dataset Exploration</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs([
            "  Rating Distribution  ",
            "  Genre Analysis  ",
            "  User Activity  "
        ])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Ratings",  f"{len(ratings):,}")
            c2.metric("Mean Rating",    f"{ratings['rating'].mean():.2f} ★")
            c3.metric("Median",         f"{ratings['rating'].median():.0f} ★")
            c4.metric("Std Dev",        f"{ratings['rating'].std():.2f}")
            st.markdown("<br>", unsafe_allow_html=True)
            dist = ratings['rating'].value_counts().sort_index()
            st.bar_chart(pd.DataFrame({'Count': dist.values}, index=[f"{int(r)}★" for r in dist.index]), color="#c89600")

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            gc = {}
            for gs in movies['genres'].dropna():
                for g in gs.split('|'):
                    gc[g] = gc.get(g, 0) + 1
            gdf = pd.DataFrame(sorted(gc.items(), key=lambda x: -x[1]), columns=['Genre', 'Count'])
            c1, c2 = st.columns(2)
            c1.metric("Unique Genres", f"{len(gc)}")
            c2.metric("Most Popular",  gdf.iloc[0]['Genre'])
            st.markdown("<br>", unsafe_allow_html=True)
            st.bar_chart(gdf.set_index('Genre'), color="#c89600")

        with tab3:
            st.markdown("<br>", unsafe_allow_html=True)
            ua = ratings.groupby('user_id').size().reset_index(name='n')
            sp = 1 - len(ratings) / (ua['user_id'].nunique() * len(movies))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Users",       f"{len(ua):,}")
            c2.metric("Avg Ratings/User",  f"{ua['n'].mean():.1f}")
            c3.metric("Max Ratings",       f"{ua['n'].max():,}")
            c4.metric("Sparsity",          f"{sp:.1%}")
            st.markdown("<br>", unsafe_allow_html=True)
            labels = ['<20', '20-50', '50-100', '100-200', '200-500', '500+']
            ua['b'] = pd.cut(ua['n'], bins=[0, 20, 50, 100, 200, 500, 10000], labels=labels)
            bc = ua['b'].value_counts().sort_index()
            st.bar_chart(pd.DataFrame({'Users': bc.values}, index=bc.index.astype(str)), color="#c89600")

    # ══════════════════════════════════════════
    # MODE 4 — Evaluation Report
    # ══════════════════════════════════════════
    elif "Evaluation" in mode:
        st.markdown('<div class="section-label">Model Evaluation Report</div>', unsafe_allow_html=True)

        rating_df, ranking_df = load_eval_data()

        # ── Run evaluation if CSVs missing
        if rating_df is None or ranking_df is None:
            st.info("No precomputed evaluation found. Running full evaluation — this may take a minute…")
            run_eval = st.button("▸ Run Evaluation Now")
            if run_eval:
                with st.spinner("Evaluating all models…"):
                    from evaluation import full_evaluation
                    rating_df, ranking_df = full_evaluation()
                st.success("Evaluation complete! Results saved to reports/")
                st.rerun()
        else:
            tab1, tab2, tab3 = st.tabs([
                "  Rating Prediction  ",
                "  Ranking Metrics  ",
                "  Model Comparison  ",
            ])

            # ── TAB 1: Rating Prediction (RMSE / MAE)
            with tab1:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">RMSE & MAE — Rating Prediction</div>', unsafe_allow_html=True)

                # Summary metrics row
                best_rmse = rating_df['rmse'].min()
                c1, c2, c3 = st.columns(3)
                c1.metric("Best RMSE",      f"{best_rmse:.4f}")
                c2.metric("Best MAE",       f"{rating_df['mae'].min():.4f}")
                c3.metric("Models Tested",  f"{len(rating_df)}")
                st.markdown("<br>", unsafe_allow_html=True)

                badges = ["eval-badge-best", "eval-badge-good", "eval-badge-base"]
                badge_labels = ["✦ Best", "Good", "Baseline"]

                for idx, row in rating_df.iterrows():
                    badge_cls   = badges[min(idx, 2)]
                    badge_label = badge_labels[min(idx, 2)]
                    is_best     = row['rmse'] == best_rmse
                    if is_best:
                        badge_cls, badge_label = "eval-badge-best", "✦ Best"

                    st.markdown(
                        f'<div class="eval-card">'
                        f'<span class="eval-badge {badge_cls}">{badge_label}</span>'
                        f'<div class="eval-model-name">{row["model"]}</div>'
                        f'<div class="eval-metric-row">'
                        f'<div class="eval-metric-box"><div class="eval-metric-val">{row["rmse"]:.4f}</div><div class="eval-metric-lbl">RMSE</div></div>'
                        f'<div class="eval-metric-box"><div class="eval-metric-val">{row["mae"]:.4f}</div><div class="eval-metric-lbl">MAE</div></div>'
                        f'<div class="eval-metric-box"><div class="eval-metric-val">{int(row["n_eval"]):,}</div><div class="eval-metric-lbl">Pairs Evaluated</div></div>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">RMSE Comparison</div>', unsafe_allow_html=True)
                max_rmse = rating_df['rmse'].max()
                for _, row in rating_df.iterrows():
                    st.markdown(compare_bar(row['model'], row['rmse'], max_rmse, "#c89600"), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">MAE Comparison</div>', unsafe_allow_html=True)
                max_mae = rating_df['mae'].max()
                for _, row in rating_df.iterrows():
                    st.markdown(compare_bar(row['model'], row['mae'], max_mae, "#888"), unsafe_allow_html=True)

                # Insight
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="insight-box">'
                    '<div class="insight-title">📌 Insight — Rating Prediction</div>'
                    '<div class="insight-text">'
                    'RMSE and MAE measure how far predicted ratings deviate from actual user ratings. '
                    'Lower is better. An RMSE around 1.0 on a 1–5 scale means predictions are off by roughly one star on average — '
                    'typical for collaborative filtering on sparse datasets like MovieLens 100K.'
                    '</div></div>',
                    unsafe_allow_html=True
                )

            # ── TAB 2: Ranking Metrics (Precision, Recall, F1)
            with tab2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Precision · Recall · F1 @ K=10</div>', unsafe_allow_html=True)

                pk_col  = [c for c in ranking_df.columns if 'precision' in c][0]
                rk_col  = [c for c in ranking_df.columns if 'recall'    in c][0]
                f1k_col = [c for c in ranking_df.columns if 'f1'        in c][0]

                best_f1 = ranking_df[f1k_col].max()

                c1, c2, c3 = st.columns(3)
                c1.metric("Best Precision@10", f"{ranking_df[pk_col].max():.4f}")
                c2.metric("Best Recall@10",    f"{ranking_df[rk_col].max():.4f}")
                c3.metric("Best F1@10",        f"{best_f1:.4f}")
                st.markdown("<br>", unsafe_allow_html=True)

                for idx, row in ranking_df.iterrows():
                    is_best     = row[f1k_col] == best_f1
                    badge_cls   = "eval-badge-best" if is_best else ("eval-badge-good" if idx == 1 else "eval-badge-base")
                    badge_label = "✦ Best F1" if is_best else ("Good" if idx == 1 else "Baseline")

                    n_users = int(row.get('n_users_eval', 0))
                    st.markdown(
                        f'<div class="eval-card">'
                        f'<span class="eval-badge {badge_cls}">{badge_label}</span>'
                        f'<div class="eval-model-name">{row["model"]}</div>'
                        f'<div class="eval-metric-row">'
                        f'<div class="eval-metric-box"><div class="eval-metric-val">{row[pk_col]:.4f}</div><div class="eval-metric-lbl">Precision@10</div></div>'
                        f'<div class="eval-metric-box"><div class="eval-metric-val">{row[rk_col]:.4f}</div><div class="eval-metric-lbl">Recall@10</div></div>'
                        f'<div class="eval-metric-box"><div class="eval-metric-val">{row[f1k_col]:.4f}</div><div class="eval-metric-lbl">F1@10</div></div>'
                        f'<div class="eval-metric-box"><div class="eval-metric-val">{n_users}</div><div class="eval-metric-lbl">Users Evaluated</div></div>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">F1@10 Comparison</div>', unsafe_allow_html=True)
                max_f1 = ranking_df[f1k_col].max()
                colors = ["#c89600", "#888", "#444"]
                for i, (_, row) in enumerate(ranking_df.iterrows()):
                    st.markdown(compare_bar(row['model'], row[f1k_col], max_f1, colors[i % 3]), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="insight-box">'
                    '<div class="insight-title">📌 Insight — Ranking Metrics</div>'
                    '<div class="insight-text">'
                    'Precision@10 measures what fraction of the top-10 recommended movies the user actually liked. '
                    'Recall@10 measures what fraction of all movies the user liked were captured in the top-10. '
                    'F1@10 is the harmonic mean of both — the best single number to compare models. '
                    'Hybrid models consistently outperform pure content-based filtering on ranking tasks.'
                    '</div></div>',
                    unsafe_allow_html=True
                )

            # ── TAB 3: Side-by-side comparison
            with tab3:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Full Model Comparison</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Rating Prediction Metrics**", unsafe_allow_html=False)
                styled_rating = rating_df.style.format({
                    'rmse': '{:.4f}', 'mae': '{:.4f}', 'n_eval': '{:,}'
                }).highlight_min(subset=['rmse', 'mae'], color='#1a1400').set_properties(**{
                    'background-color': '#111',
                    'color': '#ccc',
                    'border': '1px solid #1e1e1e',
                    'font-family': 'DM Mono, monospace',
                    'font-size': '12px',
                })
                st.dataframe(rating_df, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Ranking Metrics @ K=10**", unsafe_allow_html=False)
                st.dataframe(ranking_df, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Visual Comparison</div>', unsafe_allow_html=True)

                pk_col  = [c for c in ranking_df.columns if 'precision' in c][0]
                rk_col  = [c for c in ranking_df.columns if 'recall'    in c][0]
                f1k_col = [c for c in ranking_df.columns if 'f1'        in c][0]

                chart_df = ranking_df.set_index('model')[[pk_col, rk_col, f1k_col]]
                chart_df.columns = ['Precision@10', 'Recall@10', 'F1@10']
                st.bar_chart(chart_df, color=["#c89600", "#888", "#f0c040"])

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="insight-box">'
                    '<div class="insight-title">📌 Key Takeaway</div>'
                    '<div class="insight-text">'
                    'The Hybrid Stacking strategy achieves the highest Precision, Recall, and F1 scores, '
                    'confirming that combining content-based and collaborative signals produces better recommendations '
                    'than either approach alone. The weighted hybrid is a strong middle ground with lower compute cost.'
                    '</div></div>',
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
