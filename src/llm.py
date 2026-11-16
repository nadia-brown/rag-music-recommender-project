"""LLM steps of the RAG flow (Claude Haiku via the Anthropic SDK).

parse_request(text, genres) -> dict of target features for build_target()
explain(request, candidates) -> [{"track_id", "reason"}] grounded in candidates

The API key is read from ANTHROPIC_API_KEY (loaded from .env if present).
"""
import json
import logging
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-5-5"
MAX_PICKS = 10
FALLBACK_REASON = "Explanations unavailable right now."

UNIT_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
]
# Sanity ranges for raw-unit features (JSON schema can't express min/max)
RANGES = {"tempo": (40.0, 220.0), "loudness": (-60.0, 0.0)}
# Common spellings that don't match the dataset's genre names
GENRE_ALIASES = {"r&b": "r-n-b", "rnb": "r-n-b", "hip hop": "hip-hop", "hiphop": "hip-hop"}

log = logging.getLogger(__name__)
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def _call_json(system, user, schema, max_tokens):
    """One Messages call with JSON output. Returns parsed JSON or None.

    A missing API key is a setup error and raises instead of falling back.
    """
    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": schema},
            },
        )
    except TypeError as e:
        # The SDK raises TypeError at request time when no credentials are set
        if "authentication" in str(e):
            raise RuntimeError(
                "No Anthropic API key found. Copy .env.example to .env "
                "and set ANTHROPIC_API_KEY."
            ) from e
        raise
    except anthropic.AnthropicError as e:
        log.warning("Claude call failed: %s", e)
        return None

    if response.stop_reason != "end_turn":
        log.warning("Claude stopped early: %s", response.stop_reason)
        return None
    text = next((b.text for b in response.content if b.type == "text"), None)
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        log.warning("Claude returned malformed JSON: %r", text)
        return None


# ---------- 1. parse_request ----------

PARSE_SYSTEM = """You turn a music request into target audio features for a song recommender.

Features (use null when the request says nothing that implies a value):
- danceability, energy, valence (positivity), acousticness, instrumentalness, speechiness: 0.0 to 1.0
- tempo: beats per minute, typically 60 to 200
- loudness: average dB, typically -30 (very quiet) to -3 (very loud)
- genre: exactly one name from the allowed genre list, or null if none clearly fits

Examples:

Request: chill songs for studying late at night
{"danceability": 0.35, "energy": 0.2, "valence": 0.35, "acousticness": 0.8, "instrumentalness": 0.7, "speechiness": 0.04, "tempo": 80, "loudness": -16, "genre": "jazz"}

Request: hype workout music
{"danceability": 0.8, "energy": 0.9, "valence": 0.6, "acousticness": 0.05, "instrumentalness": 0.0, "speechiness": 0.25, "tempo": 140, "loudness": -5, "genre": "hip-hop"}

Request: sad breakup songs
{"danceability": 0.45, "energy": 0.35, "valence": 0.15, "acousticness": 0.6, "instrumentalness": 0.0, "speechiness": 0.04, "tempo": 85, "loudness": -9, "genre": null}

Request: upbeat summer road trip
{"danceability": 0.7, "energy": 0.8, "valence": 0.85, "acousticness": 0.15, "instrumentalness": 0.0, "speechiness": 0.05, "tempo": 120, "loudness": -6, "genre": "pop"}

Request: something calm to fall asleep to, no vocals
{"danceability": 0.2, "energy": 0.05, "valence": 0.2, "acousticness": 0.95, "instrumentalness": 0.9, "speechiness": 0.04, "tempo": 70, "loudness": -25, "genre": "classical"}

Request: songs that sound like a dance party
{"danceability": 0.85, "energy": 0.85, "valence": 0.7, "acousticness": 0.05, "instrumentalness": null, "speechiness": null, "tempo": 125, "loudness": -5, "genre": "electronic"}"""


def _parse_schema(genres):
    number_or_null = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    props = {f: number_or_null for f in UNIT_FEATURES + list(RANGES)}
    props["genre"] = {"anyOf": [{"type": "string", "enum": list(genres)}, {"type": "null"}]}
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def _genre_from_text(text, genres):
    """Fallback: a genre named word-for-word in the request, if any."""
    lowered = text.casefold()
    names = {g.casefold(): g for g in genres}
    for alias, name in GENRE_ALIASES.items():
        if name in genres:
            names.setdefault(alias, name)
    for name in sorted(names, key=len, reverse=True):
        if re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", lowered):
            return names[name]
    return None


def validate_features(data, genres):
    """Keep only known, well-typed features; clip to valid ranges."""
    if not isinstance(data, dict):
        return {}
    out = {}
    for key in UNIT_FEATURES + list(RANGES):
        v = data.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
            continue
        lo, hi = RANGES.get(key, (0.0, 1.0))
        out[key] = min(max(float(v), lo), hi)
    genre = data.get("genre")
    if isinstance(genre, str) and genre in genres:
        out["genre"] = genre
    return out


def parse_request(text, genres):
    """Turn a plain-English request into target features.

    Never raises on bad model output: falls back to {} (plus a genre if
    one is named in the request), which build_target treats as neutral.
    """
    genres = list(genres)
    user = (
        "Allowed genres: " + ", ".join(genres)
        + "\n\nRequest: <request>" + text + "</request>"
    )
    data = _call_json(PARSE_SYSTEM, user, _parse_schema(genres), max_tokens=4000)
    features = validate_features(data, genres)

    if data is None:
        genre = _genre_from_text(text, genres)
        if genre:
            features["genre"] = genre
    return features


# ---------- 2. explain ----------

EXPLAIN_SYSTEM = f"""You recommend songs from a fixed list of candidates retrieved for a listener's request.

Rules:
- Pick ONLY from the candidates given. Copy each track_id exactly. Never invent songs.
- Choose up to {MAX_PICKS} candidates that best fit the request, best first. Pick at least one.
- For each, write one short sentence on why it fits, based on the given data (genre, energy, mood/valence, tempo, acousticness, etc.).
- Do not state facts about a song that are not in the data."""

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "track_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["track_id", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["picks"],
    "additionalProperties": False,
}

CANDIDATE_FIELDS = [
    "track_id", "track_name", "artists", "track_genre",
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "speechiness", "tempo", "loudness",
]


def ground(picks, candidates):
    """Grounding check: keep only picks whose track_id is a candidate.

    Also drops duplicates and picks without a usable reason.
    """
    valid_ids = set(candidates["track_id"])
    seen = set()
    out = []
    for p in picks if isinstance(picks, list) else []:
        if not isinstance(p, dict):
            continue
        tid, reason = p.get("track_id"), p.get("reason")
        if tid not in valid_ids or tid in seen:
            continue
        if not isinstance(reason, str) or not reason.strip():
            continue
        seen.add(tid)
        out.append({"track_id": tid, "reason": reason.strip()})
    return out


# How far a quoted number may be from the data and still count as correct.
# Candidates are sent to Claude rounded to 2 decimals.
FACT_TOLERANCE = {"tempo": 1.0, "loudness": 0.1}
_NUMBER = r"(\d+(?:\.\d+)?)"
_SIGNED = r"(-?\d+(?:\.\d+)?)"  # only loudness is negative; "120-140 BPM" is a range


def check_reason(reason, song):
    """Fact-check the numbers Claude quotes in one explanation.

    Finds claims like "energy (0.14)", "loudness -19.05" or "100 BPM" and
    compares each to the song's real value. Returns (checked, wrong), where
    wrong is a list of (feature, claimed, actual). Claims without a number
    ("low energy") can't be checked and are ignored.
    """
    claims = []
    for feat in UNIT_FEATURES + list(RANGES):
        number = _SIGNED if feat == "loudness" else _NUMBER
        pattern = feat + r"\s*(?:of\s*|is\s*|at\s*|=\s*|:\s*)?\(?\s*" + number
        for m in re.finditer(pattern, reason, re.IGNORECASE):
            claims.append((feat, float(m.group(1))))
    for m in re.finditer(_NUMBER + r"\s*BPM", reason, re.IGNORECASE):
        claims.append(("tempo", float(m.group(1))))

    wrong = []
    for feat, claimed in claims:
        actual = float(song[feat])
        if abs(claimed - actual) > FACT_TOLERANCE.get(feat, 0.01):
            wrong.append((feat, claimed, round(actual, 2)))
    return len(claims), wrong


def _fallback_picks(candidates):
    return [
        {"track_id": tid, "reason": FALLBACK_REASON, "fallback": True}
        for tid in candidates["track_id"].head(MAX_PICKS)
    ]


def _candidates_json(candidates):
    cols = [c for c in CANDIDATE_FIELDS if c in candidates.columns]
    rows = candidates[cols].round(2).to_dict(orient="records")
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)


def explain(request, candidates):
    """Ask Claude to pick and explain songs from candidates only.

    Returns [{"track_id", "reason"}]. If the call fails or nothing survives
    the grounding check, returns the top candidates by score with
    "fallback": True and a placeholder reason.
    """
    if candidates is None or len(candidates) == 0:
        return []
    user = (
        "Request: <request>" + request + "</request>\n\n"
        "Candidates (one JSON object per line):\n" + _candidates_json(candidates)
    )
    data = _call_json(EXPLAIN_SYSTEM, user, EXPLAIN_SCHEMA, max_tokens=8000)
    picks = ground(data.get("picks") if isinstance(data, dict) else None, candidates)
    return picks or _fallback_picks(candidates)
