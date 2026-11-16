import json
from types import SimpleNamespace

import anthropic
import pandas as pd
import pytest

from src import llm

GENRES = ["classical", "hip-hop", "jazz", "pop", "r-n-b"]


@pytest.fixture
def candidates():
    return pd.DataFrame({
        "track_id": ["t1", "t2", "t3"],
        "track_name": ["One", "Two", "Three"],
        "artists": ["A", "B", "C"],
        "track_genre": ["jazz", "jazz", "pop"],
        "energy": [0.2, 0.3, 0.8],
        "score": [0.99, 0.95, 0.90],
    })


class FakeMessages:
    def __init__(self, reply=None, error=None, stop_reason="end_turn"):
        self.reply, self.error, self.stop_reason = reply, error, stop_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        text = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=text)],
        )


def use_fake(monkeypatch, **kwargs):
    fake = FakeMessages(**kwargs)
    monkeypatch.setattr(llm, "_get_client", lambda: SimpleNamespace(messages=fake))
    return fake


# ---------- grounding check ----------

def test_ground_drops_invented_ids(candidates):
    picks = [
        {"track_id": "t2", "reason": "Mellow jazz."},
        {"track_id": "made-up", "reason": "Not a candidate."},
        {"track_id": "t1", "reason": "Low energy."},
    ]
    assert llm.ground(picks, candidates) == [
        {"track_id": "t2", "reason": "Mellow jazz."},
        {"track_id": "t1", "reason": "Low energy."},
    ]


def test_ground_drops_duplicates_and_bad_entries(candidates):
    picks = [
        {"track_id": "t1", "reason": "First."},
        {"track_id": "t1", "reason": "Duplicate."},
        {"track_id": "t2", "reason": "   "},
        {"track_id": "t3"},
        "not a dict",
    ]
    assert llm.ground(picks, candidates) == [{"track_id": "t1", "reason": "First."}]
    assert llm.ground(None, candidates) == []


def test_explain_applies_grounding(monkeypatch, candidates):
    use_fake(monkeypatch, reply={"picks": [
        {"track_id": "ghost", "reason": "Invented."},
        {"track_id": "t3", "reason": "Upbeat pop."},
    ]})
    assert llm.explain("party", candidates) == [{"track_id": "t3", "reason": "Upbeat pop."}]


def test_explain_falls_back_when_nothing_is_grounded(monkeypatch, candidates):
    use_fake(monkeypatch, reply={"picks": [{"track_id": "ghost", "reason": "x"}]})
    picks = llm.explain("party", candidates)
    assert [p["track_id"] for p in picks] == ["t1", "t2", "t3"]
    assert all(p["fallback"] and p["reason"] == llm.FALLBACK_REASON for p in picks)


def test_explain_falls_back_on_api_error(monkeypatch, candidates):
    use_fake(monkeypatch, error=anthropic.APIConnectionError(request=None))
    picks = llm.explain("party", candidates)
    assert len(picks) == 3 and all(p["fallback"] for p in picks)


def test_explain_empty_candidates(candidates):
    assert llm.explain("anything", candidates.iloc[0:0]) == []


# ---------- parse_request ----------

def test_parse_request_validates_output(monkeypatch):
    fake = use_fake(monkeypatch, reply={
        "danceability": 1.4, "energy": 0.2, "valence": None, "acousticness": "high",
        "instrumentalness": 0.7, "speechiness": True, "tempo": 400, "loudness": -12,
        "genre": "jazz",
    })
    features = llm.parse_request("late night study", GENRES)
    assert features == {
        "danceability": 1.0, "energy": 0.2, "instrumentalness": 0.7,
        "tempo": 220.0, "loudness": -12.0, "genre": "jazz",
    }
    call = fake.calls[0]
    assert call["model"] == llm.MODEL == "claude-haiku-5-5"
    genre_schema = call["output_config"]["format"]["schema"]["properties"]["genre"]
    assert genre_schema["anyOf"][0]["enum"] == GENRES


def test_parse_request_drops_unknown_genre(monkeypatch):
    use_fake(monkeypatch, reply={"energy": 0.5, "genre": "polka"})
    assert llm.parse_request("polka party", GENRES) == {"energy": 0.5}


def test_parse_request_malformed_json_falls_back(monkeypatch):
    use_fake(monkeypatch, reply="{not json")
    assert llm.parse_request("something mellow", GENRES) == {}


def test_parse_request_api_error_keeps_named_genre(monkeypatch):
    use_fake(monkeypatch, error=anthropic.APIConnectionError(request=None))
    assert llm.parse_request("some R&B for tonight", GENRES) == {"genre": "r-n-b"}
    assert llm.parse_request("hip hop bangers", GENRES) == {"genre": "hip-hop"}


def test_missing_api_key_raises_clear_error(monkeypatch):
    # This is the TypeError the SDK raises at request time with no credentials
    use_fake(monkeypatch, error=TypeError("Could not resolve authentication method."))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm.parse_request("anything", GENRES)


def test_parse_request_refusal_falls_back(monkeypatch):
    use_fake(monkeypatch, reply={"energy": 0.9}, stop_reason="refusal")
    assert llm.parse_request("anything", GENRES) == {}
