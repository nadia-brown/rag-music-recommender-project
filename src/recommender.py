"""Content-based retrieval for the RAG music recommender.

load_artifacts(path)  -> {"df", "X", "scaler", "genre_cols", "feature_cols"}
build_target(features, art) -> 1-D vector with the same columns as X
retrieve(target, art, k, max_per_artist, allow_explicit) -> candidates DataFrame
"""
import math

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Already on a 0-1 scale in the dataset
AUDIO_COLS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
]
# Raw units (BPM, dB) -> scaled to 0-1 with MinMaxScaler
SCALED_COLS = ["tempo", "loudness"]
# One-hot genre columns are multiplied by this
GENRE_WEIGHT = 0.4

DISPLAY_COLS = [
    "track_id",
    "track_name",
    "artists",
    "album_name",
    "track_genre",
    "popularity",
    "explicit",
]


def _normalize(text):
    return text.fillna("").astype(str).str.normalize("NFKC").str.strip().str.casefold()


def load_artifacts(path):
    """Load the CSV, dedupe songs and build the feature matrix X."""
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Dedupe by track_name + artists, keeping the most popular version
    df["_title"] = _normalize(df["track_name"])
    df["_artists"] = _normalize(df["artists"])
    df = (
        df.sort_values("popularity", ascending=False, kind="mergesort")
        .drop_duplicates(subset=["_title", "_artists"], keep="first")
        .sort_index()
        .drop(columns=["_title", "_artists"])
        .reset_index(drop=True)
    )
    # app.py indexes results by track_id
    assert df["track_id"].is_unique, "track_id is not unique after deduping"

    # First artist in the semicolon-separated list, used for the artist cap
    df["primary_artist"] = _normalize(df["artists"].str.split(";").str[0])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[SCALED_COLS])

    genre_cols = sorted(df["track_genre"].unique().tolist())
    genres = (
        pd.get_dummies(df["track_genre"])
        .reindex(columns=genre_cols, fill_value=0)
        .to_numpy(dtype=float)
        * GENRE_WEIGHT
    )

    X = np.hstack([df[AUDIO_COLS].to_numpy(dtype=float), scaled, genres])

    return {
        "df": df,
        "X": X,
        "scaler": scaler,
        "genre_cols": genre_cols,
        "feature_cols": AUDIO_COLS + SCALED_COLS + genre_cols,
    }


def _to_float(value):
    """Return value as a finite float, or None if that isn't possible."""
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def build_target(features, art):
    """Turn parsed request features into a vector aligned with art["X"].

    features: dict like {"energy": 0.3, "tempo": 85, "loudness": -12,
    "genre": "jazz"}. tempo is in BPM and loudness in dB. Missing or
    invalid numeric values fall back to the dataset mean. A missing or
    unknown genre leaves all genre slots at 0.
    """
    features = features if isinstance(features, dict) else {}
    X = art["X"]
    means = X.mean(axis=0)
    target = np.zeros(X.shape[1])

    for i, col in enumerate(AUDIO_COLS):
        v = _to_float(features.get(col))
        target[i] = means[i] if v is None else min(max(v, 0.0), 1.0)

    scaler = art["scaler"]
    offset = len(AUDIO_COLS)
    for j, col in enumerate(SCALED_COLS):
        i = offset + j
        v = _to_float(features.get(col))
        if v is None:
            target[i] = means[i]
        else:
            scaled = v * scaler.scale_[j] + scaler.min_[j]
            target[i] = min(max(scaled, 0.0), 1.0)

    genre = features.get("genre")
    if isinstance(genre, str):
        lookup = {g.casefold(): k for k, g in enumerate(art["genre_cols"])}
        k = lookup.get(genre.strip().casefold())
        if k is not None:
            target[offset + len(SCALED_COLS) + k] = GENRE_WEIGHT

    return target


def retrieve(target, art, k=10, max_per_artist=2, allow_explicit=False):
    """Top-k songs by cosine similarity to target.

    Popularity breaks ties, at most max_per_artist songs per primary
    artist, explicit songs removed unless allow_explicit. Returns the
    display columns, raw audio features, "score" and "row" (the song's
    row in art["X"]).
    """
    df = art["df"]
    scores = cosine_similarity(np.asarray(target, dtype=float).reshape(1, -1), art["X"])[0]

    cands = df.assign(score=scores, row=np.arange(len(df)))
    if not allow_explicit:
        cands = cands[~cands["explicit"].astype(bool)]

    cands = cands.sort_values(["score", "popularity"], ascending=False, kind="mergesort")
    cands = cands[cands.groupby("primary_artist").cumcount() < max_per_artist]

    # Raw audio features are included so explain() can ground its reasons
    cols = DISPLAY_COLS + AUDIO_COLS + SCALED_COLS + ["score", "row"]
    return cands.head(k)[cols].reset_index(drop=True)
