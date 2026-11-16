
import sys
from pathlib import Path

import pandas as pd

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommender import load_artifacts, retrieve

DATA_PATH = "data/songs_5000.csv"


def evaluate_recommender(top_n=5):
    art = load_artifacts(DATA_PATH)
    df, X = art["df"], art["X"]
    results = []

    # Evaluate every song in the dataset, using its own vector as the target
    for song_index in range(len(df)):

        song = df.iloc[song_index]

        # Ask for one extra so we can drop the song itself.
        # Songs are already deduped by title + artist in load_artifacts.
        candidates = retrieve(
            X[song_index], art,
            k=top_n + 1, max_per_artist=2, allow_explicit=True
        )
        recommendations = candidates[
            candidates["row"] != song_index
        ].head(top_n)

        # Calculate same-genre recommendation rate
        same_genre_rate = (
            recommendations["track_genre"] == song["track_genre"]
        ).mean()

        # Calculate average cosine similarity
        average_similarity = recommendations["score"].mean()

        # Count distinct genres among recommendations
        unique_genres = recommendations["track_genre"].nunique()

        results.append({
            "track_genre": song["track_genre"],
            "same_genre_rate": same_genre_rate,
            "average_similarity": average_similarity,
            "unique_genres": unique_genres
        })

    results_df = pd.DataFrame(results)

    print("\nOVERALL EVALUATION")
    print("------------------------------")
    print("Songs evaluated:", len(results_df))

    print(
        "Average same-genre rate:",
        round(results_df["same_genre_rate"].mean(), 3)
    )

    print(
        "Average cosine similarity:",
        round(results_df["average_similarity"].mean(), 3)
    )

    print(
        "Average unique genres in top 5:",
        round(results_df["unique_genres"].mean(), 2)
    )

    print("\nRESULTS BY GENRE")
    print("------------------------------")

    genre_results = results_df.groupby("track_genre")[
        [
            "same_genre_rate",
            "average_similarity",
            "unique_genres"
        ]
    ].mean().round(3)

    print(genre_results)


if __name__ == "__main__":
    evaluate_recommender()
