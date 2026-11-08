import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, requirement TEXT NOT NULL, state TEXT NOT NULL,
  gaps TEXT NOT NULL DEFAULT '[]', answers TEXT NOT NULL DEFAULT '{}',
  spec TEXT, tests TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL REFERENCES projects(id),
  from_state TEXT, to_state TEXT, action TEXT NOT NULL, note TEXT DEFAULT '', at TEXT NOT NULL
);
"""
JSON_FIELDS = {"gaps", "answers", "spec", "tests"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def conn():
    c = sqlite3.connect(os.environ.get("SPECGUARD_DB", "specguard.db"))
    c.row_factory = sqlite3.Row
    try:
        c.executescript(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for f in JSON_FIELDS:
        d[f] = json.loads(d[f]) if d[f] is not None else None
    return d


def create_project(title: str, requirement: str) -> dict:
    pid = uuid.uuid4().hex[:8]
    with conn() as c:
        c.execute(
            "INSERT INTO projects (id,title,requirement,state,created_at,updated_at) VALUES (?,?,?,?,?,?)",
            (pid, title, requirement, "SUBMITTED", _now(), _now()),
        )
        c.execute(
            "INSERT INTO history (project_id,from_state,to_state,action,at) VALUES (?,?,?,?,?)",
            (pid, None, "SUBMITTED", "create", _now()),
        )
    return get_project(pid)


def get_project(pid: str) -> dict | None:
    with conn() as c:
        return _row(c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone())


def list_projects() -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT * FROM projects ORDER BY created_at DESC, rowid DESC").fetchall()
    return [_row(r) for r in rows]


def update_project(pid: str, **fields) -> None:
    sets, vals = [], []
    for k, v in fields.items():
        sets.append(f"{k}=?")
        vals.append(json.dumps(v) if k in JSON_FIELDS and v is not None else v)
    sets.append("updated_at=?")
    vals += [_now(), pid]
    with conn() as c:
        c.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id=?", vals)


def add_history(pid: str, from_state: str, to_state: str, action: str, note: str = "") -> None:
    with conn() as c:
        c.execute(
            "INSERT INTO history (project_id,from_state,to_state,action,note,at) VALUES (?,?,?,?,?,?)",
            (pid, from_state, to_state, action, note, _now()),
        )


def get_history(pid: str) -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT from_state,to_state,action,note,at FROM history WHERE project_id=? ORDER BY id", (pid,)).fetchall()
    return [dict(r) for r in rows]
