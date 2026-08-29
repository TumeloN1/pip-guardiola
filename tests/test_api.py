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

    missing = client.get("/api/players/not-a-real-id/similar")
    assert missing.status_code == 404
