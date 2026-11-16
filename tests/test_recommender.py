import numpy as np
import pandas as pd
import pytest

from src.recommender import (
    AUDIO_COLS,
    GENRE_WEIGHT,
    SCALED_COLS,
    build_target,
    load_artifacts,
    retrieve,
)


def song(i, track_name, artists, genre, *, energy=0.5, tempo=120.0,
         loudness=-8.0, popularity=50, explicit=False):
    return {
        "Unnamed: 0": i,
        "track_id": f"id{i}",
        "artists": artists,
        "album_name": "Album",
        "track_name": track_name,
        "popularity": popularity,
        "duration_ms": 200000,
        "explicit": explicit,
        "danceability": 0.5,
        "energy": energy,
        "key": 5,
        "loudness": loudness,
        "mode": 1,
        "speechiness": 0.05,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": 0.5,
        "tempo": tempo,
        "time_signature": 4,
        "track_genre": genre,
    }


@pytest.fixture
def art(tmp_path):
    rows = [
        # Artist A: four near-identical high-energy rock songs (one a collab)
        song(0, "A1", "Artist A", "rock", energy=0.90, popularity=90),
        song(1, "A2", "Artist A", "rock", energy=0.91, popularity=80),
        song(2, "A3", "Artist A;Artist B", "rock", energy=0.92, popularity=70),
        song(3, "A4", "artist a", "rock", energy=0.93, popularity=60),
        # Other rock artists, slightly further away
        song(4, "C1", "Artist C", "rock", energy=0.80, popularity=40),
        song(5, "D1", "Artist D", "rock", energy=0.80, popularity=30, explicit=True),
        song(6, "E1", "Artist E", "rock", energy=0.85, popularity=20, explicit=True),
        # Quiet jazz and pop songs; sets the tempo/loudness range
        song(7, "J1", "Artist J", "jazz", energy=0.10, tempo=60.0, loudness=-20.0),
        song(8, "P1", "Artist P", "pop", energy=0.60, tempo=180.0, loudness=-4.0),
        # Duplicate of A1 (same title + artist, different id, lower popularity)
        song(9, "a1 ", "Artist A", "rock", energy=0.90, popularity=10),
    ]
    path = tmp_path / "songs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return load_artifacts(path)


# ---------- load_artifacts ----------

def test_load_dedupes_and_keeps_most_popular(art):
    df = art["df"]
    assert len(df) == 9
    assert "id9" not in set(df["track_id"])
    assert df["track_id"].is_unique


def test_feature_matrix_shape_and_range(art):
    X = art["X"]
    assert art["genre_cols"] == ["jazz", "pop", "rock"]
    assert X.shape == (9, len(AUDIO_COLS) + len(SCALED_COLS) + 3)
    assert X.min() >= 0 and X.max() <= 1
    genre_block = X[:, -3:]
    assert set(np.unique(genre_block)) == {0.0, GENRE_WEIGHT}
    assert (genre_block.sum(axis=1) == pytest.approx(GENRE_WEIGHT))


# ---------- build_target ----------

def test_build_target_matches_matrix_columns(art):
    target = build_target({"energy": 0.9, "genre": "rock"}, art)
    assert target.shape == (art["X"].shape[1],)
    rock = len(AUDIO_COLS) + len(SCALED_COLS) + art["genre_cols"].index("rock")
    assert target[rock] == GENRE_WEIGHT
    assert target[-3:].sum() == pytest.approx(GENRE_WEIGHT)


def test_build_target_scales_tempo_and_loudness(art):
    t_idx = len(AUDIO_COLS) + SCALED_COLS.index("tempo")
    l_idx = len(AUDIO_COLS) + SCALED_COLS.index("loudness")
    # Dataset range: tempo 60-180 BPM, loudness -20 to -4 dB
    assert build_target({"tempo": 60}, art)[t_idx] == pytest.approx(0.0)
    assert build_target({"tempo": 120}, art)[t_idx] == pytest.approx(0.5)
    assert build_target({"tempo": 400}, art)[t_idx] == 1.0          # clipped
    assert build_target({"loudness": -4}, art)[l_idx] == pytest.approx(1.0)


def test_build_target_fallbacks(art):
    means = art["X"].mean(axis=0)
    e_idx = AUDIO_COLS.index("energy")
    target = build_target({"energy": "loud", "genre": "polka"}, art)
    assert target[e_idx] == pytest.approx(means[e_idx])   # invalid -> mean
    assert target[-3:].sum() == 0                         # unknown genre -> zeros
    assert build_target({"energy": 1.7}, art)[e_idx] == 1.0   # clipped
    assert build_target(None, art) == pytest.approx(means[:-3].tolist() + [0, 0, 0])


def test_build_target_genre_is_case_insensitive(art):
    assert build_target({"genre": " Rock "}, art)[-3:].sum() == pytest.approx(GENRE_WEIGHT)


# ---------- retrieve ----------

def test_retrieve_columns_and_row_alignment(art):
    target = art["X"][0]
    res = retrieve(target, art, k=5, allow_explicit=True)
    for col in ["track_id", "track_name", "artists", "score", "row"]:
        assert col in res.columns
    assert res.iloc[0]["track_id"] == "id0"
    assert res.iloc[0]["score"] == pytest.approx(1.0)
    assert res["score"].is_monotonic_decreasing
    for _, r in res.iterrows():
        assert art["df"].iloc[int(r["row"])]["track_id"] == r["track_id"]


def test_retrieve_artist_cap_uses_first_artist(art):
    target = build_target({"energy": 0.91, "genre": "rock"}, art)
    res = retrieve(target, art, k=10, max_per_artist=2, allow_explicit=True)
    first = res["artists"].str.split(";").str[0].str.strip().str.casefold()
    assert (first == "artist a").sum() == 2
    assert first.value_counts().max() <= 2
    # Cap doesn't stop us filling k from other artists
    assert len(res) == 7  # 2 from Artist A + C, D, E, J, P


def test_retrieve_explicit_filter(art):
    target = build_target({"energy": 0.82, "genre": "rock"}, art)
    blocked = retrieve(target, art, k=10, allow_explicit=False)
    assert not blocked["explicit"].any()
    assert {"id5", "id6"}.isdisjoint(blocked["track_id"])

    allowed = retrieve(target, art, k=10, allow_explicit=True)
    assert {"id5", "id6"} <= set(allowed["track_id"])


def test_retrieve_popularity_breaks_ties(art):
    # Same features; C1 (popularity 40) beats D1 (popularity 30)
    target = art["X"][4]
    res = retrieve(target, art, k=10, allow_explicit=True)
    ids = list(res["track_id"])
    assert ids.index("id4") < ids.index("id5")


def test_retrieve_respects_k(art):
    res = retrieve(art["X"][0], art, k=3, allow_explicit=True)
    assert len(res) == 3
