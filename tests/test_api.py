"""API smoke tests against the in-process FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient

from kindred.api import app


def test_health_and_kdb_similar():
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ok"] is True
    assert body["n_outfield"] > 1000

    search = client.get("/api/players", params={"q": "De Bruyne"})
    assert search.status_code == 200
    ids = {row["id"] for row in search.json()}
    assert "e46012d4-2020" in ids

    sim = client.get(
        "/api/players/e46012d4-2020/similar",
        params={"comps": "Premier League", "k": 5, "min_minutes": 900},
    )
    assert sim.status_code == 200
    payload = sim.json()
    assert payload["query"]["player"] == "Kevin De Bruyne"
    assert len(payload["results"]) == 5
    assert payload["results"][0]["player_id"] != "e46012d4-2020"
    assert payload["results"][0]["fbref_id"]

    missing = client.get("/api/players/not-a-real-id/similar")
    assert missing.status_code == 404

    gone = client.get("/api/players/not-a-real-id")
    assert gone.status_code == 404

    profile = client.get("/api/players/e46012d4-2020/profile")
    assert profile.status_code == 200
    arch = profile.json()["archetypes"]
    assert arch
    for row in arch:
        assert " " in row["name"] or row["name"][0].isupper()
        assert "_" not in row["name"]
        assert "/" not in row["name"]
        assert "p90" not in row["name"].lower()
        assert "blurb" in row
        assert row["blurb"]

    headline = profile.json()["headline"]
    labels = {row["label"] for row in headline}
    assert "Goals" in labels
    assert "Assists" in labels
    goals = next(row["value"] for row in headline if row["label"] == "Goals")
    assists = next(row["value"] for row in headline if row["label"] == "Assists")
    assert goals == "13"
    assert assists == "20"

    search = client.get("/api/players", params={"q": "de bruyne"})
    ids = [row["id"] for row in search.json()]
    assert ids and ids[0].startswith("e46012d4-")
