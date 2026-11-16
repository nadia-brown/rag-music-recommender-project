
import pandas as pd

# ------------------------------------------
# 1. Load the original Spotify dataset
# ------------------------------------------

df = pd.read_csv("data/dataset.csv")

# ------------------------------------------
# 2. Select our 10 genres
# ------------------------------------------

selected_genres = [
    "pop",
    "hip-hop",
    "r-n-b",
    "rock",
    "jazz",
    "electronic",
    "country",
    "soul",
    "reggae",
    "classical"
]

df = df[df["track_genre"].isin(selected_genres)].copy()

# ------------------------------------------
# 3. Remove duplicate Spotify track IDs
# ------------------------------------------

df = df.drop_duplicates(subset=["track_id"])

# ------------------------------------------
# 4. Check available songs per genre
# ------------------------------------------

print("\nAvailable unique tracks per genre:")
print(df["track_genre"].value_counts().sort_index())

# ------------------------------------------
# 5. Sample up to 500 songs per genre
# ------------------------------------------

songs_per_genre = 500

sampled_df = (
    df.groupby("track_genre", group_keys=False)
    .apply(
        lambda group: group.sample(
            n=min(len(group), songs_per_genre),
            random_state=42
        ),
        include_groups=False
    )
    .reset_index(drop=True)
)

# Restore the genre column because
# include_groups=False excludes grouping columns.
# Instead, use the groupby sampling approach below
# for the final dataset.
sampled_df = (
    df.groupby("track_genre", group_keys=False)
    .sample(
        n=songs_per_genre,
        random_state=42
    )
    if df["track_genre"].value_counts().min() >= songs_per_genre
    else df.groupby("track_genre", group_keys=False)
    .apply(
        lambda group: group.sample(
            n=min(len(group), songs_per_genre),
            random_state=42
        )
    )
)

sampled_df = sampled_df.reset_index(drop=True)

# ------------------------------------------
# 6. Save the expanded dataset
# ------------------------------------------

sampled_df.to_csv("data/songs_5000.csv", index=False)

print("\nExpanded dataset shape:", sampled_df.shape)

print("\nSongs per genre:")
print(sampled_df["track_genre"].value_counts().sort_index())

print("\nDuplicate track IDs:",
      sampled_df["track_id"].duplicated().sum())

print("\nMissing values:")
print(sampled_df.isnull().sum())
