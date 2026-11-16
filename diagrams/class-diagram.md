# Class Diagram: Modules and Dependencies

See [architecture.md](architecture.md) for the system overview.

Each module is drawn as a class with its public functions and main
constants. `llm.py` does not import `recommender.py`. `app.py` passes
the candidates DataFrame from `retrieve()` to `explain()`.

```mermaid
classDiagram
    direction LR

    class app
    <<module>> app
    app : +DATA_PATH str
    app : +TASTE_WEIGHT float
    app : +NUDGE float
    app : +get_artifacts() dict
    app : +update_taste(row, liked) None

    class recommender
    <<module>> recommender
    recommender : +AUDIO_COLS list
    recommender : +SCALED_COLS list
    recommender : +GENRE_WEIGHT float
    recommender : +load_artifacts(path) dict
    recommender : +build_target(features, art) ndarray
    recommender : +retrieve(target, art, k, max_per_artist, allow_explicit) DataFrame

    class llm
    <<module>> llm
    llm : +MODEL str
    llm : +FALLBACK_REASON str
    llm : +parse_request(text, genres) dict
    llm : +validate_features(data, genres) dict
    llm : +explain(request, candidates) list
    llm : +ground(picks, candidates) list

    class anthropic
    <<external>> anthropic
    anthropic : +Anthropic()
    anthropic : +messages.create()

    class streamlit
    <<external>> streamlit

    class sklearn
    <<external>> sklearn
    sklearn : +MinMaxScaler
    sklearn : +cosine_similarity()

    class songs_csv
    <<data>> songs_csv

    app ..> recommender : imports load_artifacts, build_target, retrieve
    app ..> llm : imports parse_request, explain
    app ..> streamlit : UI and session_state
    llm ..> anthropic : messages.create
    recommender ..> sklearn : scaling and similarity
    recommender ..> songs_csv : reads in load_artifacts
    note for app "app.py"
    note for recommender "src/recommender.py"
    note for llm "src/llm.py"
    note for songs_csv "data/songs_5000.csv"
```
