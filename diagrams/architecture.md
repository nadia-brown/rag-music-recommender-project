# Architecture

System overview of the RAG music recommender. Names match the code in
`app.py`, `src/recommender.py`, `src/llm.py`, `tests/` and `scripts/`.

More detailed diagrams:

- [Sequence diagram](sequence-diagram.md): one request, step by step,
  including the fallback path and like/skip feedback.
- [Class diagram](class-diagram.md): each module's public functions and
  the dependencies between them.

## System overview (components, data flow, checks)

Data moves top to bottom: **input** (the user's request and the song
dataset), **process** (parse, retrieve, explain, ground), then **output**
(song cards in the app). Dashed lines are calls to Claude. Dotted boxes on
the side show where results are checked:

- **Automated checks in the app:** `validate_features()` rejects
  out-of-range features or genres not in the dataset. `ground()` drops any
  song Claude names that isn't in the retrieved candidates.
- **Human checks:** the listener reads each reason and likes or skips the
  song, which updates the taste profile for the next request. The
  developer also reviews real Claude output during manual live testing.
  Nothing automated scores how good the explanations are, so this human
  review is the only check on their quality.
- **Testing:** pytest unit tests (Claude is replaced by a fake client) and
  `scripts/evaluate.py`, which scores retrieval on its own.

```mermaid
flowchart TD
    subgraph INPUT["1. Input"]
        REQ(["User request in app.py<br/>e.g. chill songs for studying"])
        CSV[("data/songs_5000.csv")]
    end

    subgraph PROCESS["2. Process"]
        LOAD["load_artifacts()<br/>dedupe, scale tempo and loudness,<br/>one-hot genre x 0.4"]
        PARSE["Request parser<br/>llm.parse_request()"]
        VAL{"Validator<br/>llm.validate_features()<br/>values in range? genre in dataset?"}
        TARGET["recommender.build_target()<br/>features to target vector"]
        BLEND["Blend<br/>0.7 x target + 0.3 x taste"]
        RET["Retriever<br/>recommender.retrieve()<br/>cosine similarity, top 10,<br/>max 2 per artist, explicit filter"]
        EXP["Explainer<br/>llm.explain()"]
        GROUND{"Grounding check<br/>llm.ground()<br/>track_id in candidates?"}
        FB["Fallback picks<br/>top candidates, fallback True,<br/>Explanations unavailable right now"]
    end

    CLAUDE(("Claude API<br/>claude-haiku-5-5"))

    subgraph OUTPUT["3. Output"]
        UI[/"Song cards in app.py<br/>title, artist, genre, match score, reason"/]
    end

    subgraph HUMAN["Human in the loop"]
        FEED["Listener likes or skips a song<br/>update_taste()"]
        TASTE[("st.session_state.taste")]
        DEV["Developer live testing<br/>reviews real Claude output"]
    end

    subgraph TESTING["Testing and evaluation"]
        UT["Unit tests, pytest<br/>tests/test_recommender.py<br/>tests/test_llm.py with a fake Claude client"]
        EVAL["Evaluator<br/>scripts/evaluate.py<br/>same-genre rate, average similarity"]
    end

    REQ --> PARSE
    PARSE -. "prompt with 6 examples" .-> CLAUDE
    CLAUDE -. "JSON features" .-> VAL
    VAL -->|"valid features, or neutral fallback"| TARGET
    CSV --> LOAD
    LOAD -->|"feature matrix X"| RET
    TARGET --> BLEND
    TASTE --> BLEND
    BLEND --> RET
    RET -->|"10 candidates"| EXP
    EXP -. "candidates only" .-> CLAUDE
    CLAUDE -. "picks with reasons" .-> GROUND
    GROUND -->|"grounded picks"| UI
    GROUND -->|"none left or API failed"| FB
    FB --> UI
    UI --> FEED
    FEED --> TASTE
    DEV -.->|"reviews"| UI

    UT -.->|"tests"| VAL
    UT -.->|"tests"| TARGET
    UT -.->|"tests"| RET
    UT -.->|"tests"| GROUND
    EVAL -.->|"measures"| RET

    classDef ai fill:#ede7f6,stroke:#5e35b1,color:#1a1a1a
    classDef check fill:#fff3e0,stroke:#ef6c00,color:#1a1a1a
    classDef human fill:#e8f5e9,stroke:#2e7d32,color:#1a1a1a
    classDef test fill:#e3f2fd,stroke:#1565c0,color:#1a1a1a
    class CLAUDE,PARSE,EXP ai
    class VAL,GROUND,FB check
    class FEED,TASTE,DEV human
    class UT,EVAL test
```

Colors: purple is the AI step, orange is an automated check, green is a
human, blue is testing.
