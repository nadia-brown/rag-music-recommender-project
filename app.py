"""Streamlit UI for rag-music-recommender.

Run from the project root:  streamlit run app.py
"""
import numpy as np
import streamlit as st

from src.recommender import load_artifacts, build_target, retrieve
from src.llm import parse_request, explain

DATA_PATH = "data/songs_5000.csv"
TASTE_WEIGHT = 0.3   # how much the taste profile pulls on each new request
NUDGE = 0.2          # how far one like/skip moves the taste profile

st.set_page_config(page_title="Music Recommender", page_icon="🎵")


# ---------- 1. Load data once (cached across reruns) ----------
@st.cache_resource
def get_artifacts():
    # Expected keys: df, X (feature matrix), scaler, genre_cols
    return load_artifacts(DATA_PATH)


art = get_artifacts()
df, X = art["df"], art["X"]


# ---------- 2. State that must survive reruns ----------
defaults = {"taste": None, "results": None, "request": "", "feedback": {}}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)


def update_taste(row: int, liked: bool):
    """Move the taste vector toward (like) or away from (skip) a song."""
    song_vec = X[row]
    taste = st.session_state.taste
    if taste is None:
        taste = song_vec.copy() if liked else np.full_like(song_vec, 0.5)
    direction = 1 if liked else -1
    taste = taste + direction * NUDGE * (song_vec - taste)
    st.session_state.taste = np.clip(taste, 0, 1)
    st.session_state.feedback[row] = "👍" if liked else "👎"


# ---------- 3. UI: request form ----------
st.title("🎵 Music Recommender")
st.caption("Describe what you want to hear. Like or skip songs to tune your taste profile.")

with st.sidebar:
    k = st.slider("Number of songs", 3, 10, 5)
    allow_explicit = st.toggle("Allow explicit songs", value=True)
    if st.button("Reset taste profile"):
        st.session_state.taste = None
        st.session_state.feedback = {}
    st.write("Taste profile:", "active" if st.session_state.taste is not None else "not set yet")

# st.form = the LLM only runs when the user clicks Submit, not on every rerun
with st.form("request_form"):
    request = st.text_input("What are you in the mood for?",
                            placeholder="chill songs for studying late at night")
    submitted = st.form_submit_button("Recommend")


# ---------- 4. Pipeline: parse -> target -> retrieve -> explain ----------
if submitted and request.strip():
    with st.spinner("Finding songs..."):
        try:
            # LLM call 1: plain English -> {"energy": 0.3, ..., "genre": "..."}
            features = parse_request(request, genres=art["genre_cols"])
            target = build_target(features, art)

            # Blend in the taste profile if the user has given feedback
            if st.session_state.taste is not None:
                target = (1 - TASTE_WEIGHT) * target + TASTE_WEIGHT * st.session_state.taste

            # Retrieval: cosine similarity, artist cap, explicit filter
            candidates = retrieve(target, art, k=10, max_per_artist=2,
                                  allow_explicit=allow_explicit)

            # LLM call 2: grounded explanations, returns [{"track_id", "reason"}, ...]
            picks = explain(request, candidates)

            # Grounding check: drop anything the LLM invented
            valid_ids = set(candidates["track_id"])
            picks = [p for p in picks if p.get("track_id") in valid_ids][:k]

            st.session_state.results = (candidates, picks)
            st.session_state.request = request
        except Exception as e:
            st.error(f"Something went wrong: {e}")


# ---------- 5. Display results + feedback ----------
if st.session_state.results:
    candidates, picks = st.session_state.results
    by_id = candidates.set_index("track_id")
    st.subheader(f"For: “{st.session_state.request}”")

    if not picks:
        st.info("No matches came back. Try rephrasing your request.")

    for p in picks:
        song = by_id.loc[p["track_id"]]
        row = int(song["row"])   # position of this song in X
        with st.container(border=True):
            st.markdown(f"**{song['track_name']}** — {song['artists']}")
            st.caption(f"{song['track_genre']} · match {song['score']:.2f}")
            st.write(p["reason"])

            col1, col2, col3 = st.columns([1, 1, 4])
            col1.button("👍", key=f"like_{row}", on_click=update_taste, args=(row, True))
            col2.button("👎", key=f"skip_{row}", on_click=update_taste, args=(row, False))
            if row in st.session_state.feedback:
                col3.write(st.session_state.feedback[row])