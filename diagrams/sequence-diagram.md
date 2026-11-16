# Sequence Diagram: One Request

Names match the code in `app.py`, `src/llm.py` and `src/recommender.py`.
See [architecture.md](architecture.md) for the system overview.

`load_artifacts` runs once at startup and is cached with
`st.cache_resource`. Each request then goes through two Claude calls with
local retrieval in between. Like/skip feedback only updates
`st.session_state.taste` and makes no LLM call.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant App as app.py
    participant LLM as src/llm.py
    participant API as Claude API (claude-haiku-5-5)
    participant Rec as src/recommender.py

    Note over App,Rec: Startup, once: get_artifacts() calls load_artifacts(DATA_PATH), cached by st.cache_resource

    User->>App: Types a request and clicks Recommend
    App->>LLM: parse_request(request, genres=genre_cols)
    LLM->>API: messages.create with PARSE_SYSTEM and a JSON schema
    alt Call succeeds
        API-->>LLM: JSON target features
        LLM->>LLM: validate_features(data, genres) clips values and drops unknown genre
    else API error, refusal or malformed JSON
        API-->>LLM: error or unusable reply
        LLM->>LLM: Fall back to neutral features plus any genre named in the request
    end
    LLM-->>App: features dict

    App->>Rec: build_target(features, art)
    Rec-->>App: target vector, same columns as X
    opt st.session_state.taste is set
        App->>App: target = 0.7 x target + 0.3 x taste
    end
    App->>Rec: retrieve(target, art, k=10, max_per_artist=2, allow_explicit)
    Rec-->>App: candidates DataFrame with score and row

    App->>LLM: explain(request, candidates)
    LLM->>API: messages.create with EXPLAIN_SYSTEM and the candidate rows
    alt Call succeeds
        API-->>LLM: picks as track_id and reason
    else API error, refusal or malformed JSON
        API-->>LLM: error or unusable reply
    end
    LLM->>LLM: ground(picks, candidates) drops track_ids not in candidates
    opt No grounded picks left
        LLM->>LLM: Fallback picks, top candidates with fallback True
    end
    LLM-->>App: list of track_id and reason

    App->>App: Keep picks whose track_id is in candidates, first k
    App->>App: st.session_state.results = (candidates, picks)
    App-->>User: Song cards with reasons and like and skip buttons

    Note over App: Any exception, such as a missing API key, is shown with st.error

    User->>App: Clicks like or skip on a song
    App->>App: update_taste(row, liked) moves st.session_state.taste toward or away from X[row]
    Note over App,API: No LLM call. The taste vector is blended into the next request.
```
