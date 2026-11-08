import pytest
from fastapi.testclient import TestClient

from app.main import app

VAGUE = "Create a login page."
COMPLETE = ("As an admin user, the system must reject invalid passwords shorter than 8 characters "
            "with an error message when the user is authorized via token, and show the result within 2 seconds.")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECGUARD_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("SPECGUARD_MODE", "heuristic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return TestClient(app)


def make(client, text=VAGUE):
    r = client.post("/api/projects", json={"title": "Demo", "requirement": text})
    assert r.status_code == 201
    return r.json()


def answer_all(client, pid, p):
    answers = {g["id"]: f"answer for {g['id']}" for g in p["open_blocking"]}
    r = client.post(f"/api/projects/{pid}/answers", json={"answers": answers})
    assert r.status_code == 200
    return r.json()


def test_new_project_starts_submitted(client):
    p = make(client)
    assert p["state"] == "SUBMITTED"
    assert p["history"][0]["action"] == "create"


def test_rejects_too_short_requirement(client):
    assert client.post("/api/projects", json={"title": "x", "requirement": "hi"}).status_code == 422


def test_vague_requirement_needs_clarification(client):
    p = make(client)
    p = client.post(f"/api/projects/{p['id']}/analyze").json()
    assert p["state"] == "NEEDS_CLARIFICATION"
    assert {"acceptance", "validation", "security"} <= {g["id"] for g in p["open_blocking"]}
    assert p["spec"] is None


def test_cannot_skip_stages(client):
    p = make(client)
    pid = p["id"]
    assert client.post(f"/api/projects/{pid}/approve", json={}).status_code == 409
    assert client.post(f"/api/projects/{pid}/tests").status_code == 409
    client.post(f"/api/projects/{pid}/analyze")
    assert client.post(f"/api/projects/{pid}/approve", json={}).status_code == 409


def test_reanalysis_blocked_until_questions_answered(client):
    p = make(client)
    pid = p["id"]
    client.post(f"/api/projects/{pid}/analyze")
    again = client.post(f"/api/projects/{pid}/analyze").json()
    assert again["state"] == "NEEDS_CLARIFICATION"


def test_unknown_question_id_rejected(client):
    p = make(client)
    client.post(f"/api/projects/{p['id']}/analyze")
    r = client.post(f"/api/projects/{p['id']}/answers", json={"answers": {"nope": "something"}})
    assert r.status_code == 400


def test_full_happy_path(client):
    p = make(client)
    pid = p["id"]
    p = client.post(f"/api/projects/{pid}/analyze").json()
    p = answer_all(client, pid, p)
    p = client.post(f"/api/projects/{pid}/analyze").json()
    assert p["state"] == "SPEC_READY" and p["spec"]["criteria"]
    p = client.post(f"/api/projects/{pid}/approve", json={"note": "looks right"}).json()
    assert p["state"] == "APPROVED"
    p = client.post(f"/api/projects/{pid}/tests").json()
    assert p["state"] == "TESTS_READY"
    req_ids = {r["id"] for r in p["spec"]["requirements"]}
    crit_ids = {c["id"] for c in p["spec"]["criteria"]}
    assert p["tests"] and all(t["requirement_id"] in req_ids and t["criterion_id"] in crit_ids for t in p["tests"])
    assert all(t["executed"] is False for t in p["tests"])
    assert [h["action"] for h in p["history"]][-3:] == ["analyze", "approve", "generate_tests"]


def test_complete_requirement_goes_straight_to_spec_ready(client):
    p = make(client, COMPLETE)
    p = client.post(f"/api/projects/{p['id']}/analyze").json()
    assert p["state"] == "SPEC_READY"


def test_revising_requirement_discards_approval_and_tests(client):
    p = make(client, COMPLETE)
    pid = p["id"]
    client.post(f"/api/projects/{pid}/analyze")
    client.post(f"/api/projects/{pid}/approve", json={})
    client.post(f"/api/projects/{pid}/tests")
    p = client.put(f"/api/projects/{pid}/requirement", json={"requirement": "Something quite different now."}).json()
    assert p["state"] == "SUBMITTED"
    assert p["spec"] is None and p["tests"] is None and p["answers"] == {}


def test_coverage_never_reports_generated_tests_as_executed(client):
    p = make(client, COMPLETE)
    pid = p["id"]
    client.post(f"/api/projects/{pid}/analyze")
    client.post(f"/api/projects/{pid}/approve", json={})
    client.post(f"/api/projects/{pid}/tests")
    rows = client.get(f"/api/projects/{pid}/coverage").json()["rows"]
    assert rows and all(r["tests_executed"] == 0 and r["status"] == "generated, not executed" for r in rows)


def test_export_contains_spec_and_tests(client):
    p = make(client, COMPLETE)
    pid = p["id"]
    client.post(f"/api/projects/{pid}/analyze")
    client.post(f"/api/projects/{pid}/approve", json={})
    client.post(f"/api/projects/{pid}/tests")
    md = client.get(f"/api/projects/{pid}/export").text
    assert "## Requirements" in md and "Generated tests (not executed)" in md


def test_missing_project_is_404(client):
    assert client.get("/api/projects/nope").status_code == 404
