from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from . import db
from .agents import LLMError, get_backend
from .models import AnswersIn, ApproveIn, Gap, ProjectCreate, RequirementUpdate, Spec
from .workflow import State, WorkflowError, allowed_actions, require, state_after_analysis

app = FastAPI(title="SpecGuard", version="0.1.0")
STATIC = Path(__file__).parent / "static"


@app.exception_handler(WorkflowError)
async def _workflow_error(_, exc: WorkflowError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def _llm_error(_, exc: LLMError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


def _load(pid: str) -> dict:
    p = db.get_project(pid)
    if p is None:
        raise HTTPException(404, "Project not found")
    return p


def _view(p: dict) -> dict:
    gaps = [Gap(**g) for g in p["gaps"]]
    open_blocking = [g.model_dump() for g in gaps if g.severity == "blocking" and not p["answers"].get(g.id, "").strip()]
    keys = ("id", "title", "requirement", "state", "answers", "spec", "tests", "created_at", "updated_at")
    return {**{k: p[k] for k in keys}, "gaps": [g.model_dump() for g in gaps], "open_blocking": open_blocking,
            "allowed_actions": allowed_actions(State(p["state"])), "history": db.get_history(p["id"])}


def _transition(p: dict, new_state: State, action: str, note: str = "", **fields) -> dict:
    db.update_project(p["id"], state=new_state.value, **fields)
    db.add_history(p["id"], p["state"], new_state.value, action, note)
    return _view(db.get_project(p["id"]))


@app.get("/api/health")
def health():
    b = get_backend()
    return {"status": "ok", "mode": b.name, "label": getattr(b, "label", b.name)}


@app.post("/api/projects", status_code=201)
def create_project(body: ProjectCreate):
    return _view(db.create_project(body.title.strip(), body.requirement.strip()))


@app.get("/api/projects")
def list_projects():
    return [{"id": p["id"], "title": p["title"], "state": p["state"], "updated_at": p["updated_at"]} for p in db.list_projects()]


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return _view(_load(pid))


@app.put("/api/projects/{pid}/requirement")
def revise(pid: str, body: RequirementUpdate):
    p = _load(pid)
    require("revise", State(p["state"]))
    # Editing the requirement invalidates everything derived from it, including any approval.
    return _transition(p, State.SUBMITTED, "revise", "Requirement edited; analysis, spec, approval and tests were reset",
                       requirement=body.requirement.strip(), gaps=[], answers={}, spec=None, tests=None)


@app.post("/api/projects/{pid}/analyze")
def analyze(pid: str, ):
    p = _load(pid)
    require("analyze", State(p["state"]))
    backend = get_backend()
    found = backend.find_gaps(p["requirement"], p["answers"])
    merged = {g["id"]: g for g in p["gaps"]}
    for g in found:
        merged.setdefault(g.id, g.model_dump())
    gaps = [Gap(**g) for g in merged.values()]
    open_blocking = [g for g in gaps if g.severity == "blocking" and not p["answers"].get(g.id, "").strip()]
    new_state = state_after_analysis(len(open_blocking))
    spec = backend.build_spec(p["title"], p["requirement"], p["answers"], gaps).model_dump() if new_state is State.SPEC_READY else None
    note = f"{len(open_blocking)} blocking question(s) open" if open_blocking else "No blocking gaps; spec generated"
    return _transition(p, new_state, "analyze", f"{note} [{backend.name}]", gaps=[g.model_dump() for g in gaps], spec=spec)


@app.post("/api/projects/{pid}/answers")
def answer(pid: str, body: AnswersIn):
    p = _load(pid)
    require("answer", State(p["state"]))
    known = {g["id"] for g in p["gaps"]}
    unknown = sorted(set(body.answers) - known)
    if unknown:
        raise HTTPException(400, f"Unknown question id(s): {', '.join(unknown)}")
    answers = dict(p["answers"])
    for gid, text in body.answers.items():
        text = text.strip()
        if text and len(text) < 3:
            raise HTTPException(400, f"Answer for '{gid}' is too short to be useful")
        if text:
            answers[gid] = text
    return _transition(p, State.NEEDS_CLARIFICATION, "answer", f"{len(body.answers)} answer(s) saved", answers=answers)


@app.post("/api/projects/{pid}/approve")
def approve(pid: str, body: ApproveIn):
    p = _load(pid)
    require("approve", State(p["state"]))
    return _transition(p, State.APPROVED, "approve", body.note.strip())


@app.post("/api/projects/{pid}/tests")
def generate_tests(pid: str):
    p = _load(pid)
    require("generate_tests", State(p["state"]))
    spec = Spec(**p["spec"])
    suite = get_backend().build_tests(spec)
    req_ids, crit_ids = {r.id for r in spec.requirements}, {c.id for c in spec.criteria}
    bad = [t.id for t in suite.tests if t.requirement_id not in req_ids or t.criterion_id not in crit_ids]
    if bad:  # never trust generated output blindly
        raise LLMError(f"Generated tests reference unknown requirements or criteria: {', '.join(bad)}")
    if not suite.tests:
        raise LLMError("No tests were generated")
    return _transition(p, State.TESTS_READY, "generate_tests", f"{len(suite.tests)} tests generated, none executed",
                       tests=[t.model_dump() for t in suite.tests])


@app.get("/api/projects/{pid}/coverage")
def coverage(pid: str):
    p = _load(pid)
    if not p["spec"]:
        raise HTTPException(409, "No spec yet")
    tests = p["tests"] or []
    rows = []
    for r in p["spec"]["requirements"]:
        ts = [t for t in tests if t["requirement_id"] == r["id"]]
        executed = sum(1 for t in ts if t["executed"])
        status = "no tests" if not ts else ("executed" if executed == len(ts) else "generated, not executed")
        rows.append({"requirement_id": r["id"], "text": r["text"], "tests_generated": len(ts), "tests_executed": executed, "status": status})
    return {"rows": rows}


@app.get("/api/projects/{pid}/export", response_class=PlainTextResponse)
def export(pid: str):
    p = _load(pid)
    if not p["spec"]:
        raise HTTPException(409, "No spec yet")
    s = p["spec"]
    out = [f"# {p['title']}", "", s["summary"], "", "## Requirements"]
    out += [f"- **{r['id']}**: {r['text']}" for r in s["requirements"]]
    out += ["", "## Acceptance criteria"]
    out += [f"- **{c['id']}** ({c['requirement_id']}, {c['category']}): {c['text']}" for c in s["criteria"]]
    if s["assumptions"]:
        out += ["", "## Assumptions"] + [f"- {a}" for a in s["assumptions"]]
    if p["tests"]:
        out += ["", "## Generated tests (not executed)"]
        out += [f"- **{t['id']}** [{t['kind']}] {t['title']} -> {t['requirement_id']}/{t['criterion_id']}" for t in p["tests"]]
    return "\n".join(out) + "\n"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
