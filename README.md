# SpecGuard

Check a software requirement *before* anyone writes code for it. SpecGuard finds what is missing or ambiguous, asks targeted questions, produces a spec with acceptance criteria, waits for human approval, and only then generates test cases linked back to each requirement.

## Why it is more than an AI wrapper
- **The workflow is enforced in code** (`app/workflow.py`), not in prompts. A project cannot be approved until its blocking questions are answered, and tests cannot be generated until a human approves the spec.
- **Editing the requirement discards the approval**, the spec, and the tests.
- **Model output is never trusted.** Every LLM response is validated with pydantic, retried once on failure, and generated tests must reference real requirement and criterion ids.
- **Honest coverage.** Generated tests are marked `executed: false`. The coverage view never reports a requirement as verified.
- **Auditable.** Every transition is recorded in a history table.

## Run it
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload        # http://localhost:8000
pytest -q                            # unit + API tests
python -m eval.run_eval              # gap-detection evaluation
```
Docker: `docker build -t specguard . && docker run -p 8000:8000 -v specguard-data:/data specguard`

### Two analysis modes
| Mode | When | Notes |
|---|---|---|
| `heuristic` | default, no API key | Deterministic keyword checklist. Good for demos, tests, CI. It checks for the presence of words, not meaning. |
| `llm` | `ANTHROPIC_API_KEY` set | Claude finds gaps, writes the spec and tests. Model set via `SPECGUARD_MODEL`. |

## Workflow
```
SUBMITTED -> NEEDS_CLARIFICATION -> SPEC_READY -> APPROVED -> TESTS_READY
                 ^   answer + re-analyze  |
                 +------------------------+          any state --revise--> SUBMITTED
```

## API
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/projects` | Create a project from a requirement |
| POST | `/api/projects/{id}/analyze` | Find gaps; generates a spec when none are blocking |
| POST | `/api/projects/{id}/answers` | Answer open questions |
| POST | `/api/projects/{id}/approve` | Human approval (only from SPEC_READY) |
| POST | `/api/projects/{id}/tests` | Generate tests (only from APPROVED) |
| PUT | `/api/projects/{id}/requirement` | Revise; resets the project |
| GET | `/api/projects/{id}/coverage` | Requirement-to-test traceability |
| GET | `/api/projects/{id}/export` | Markdown export |

## Evaluation
`eval/cases.json` is a **starter set of 12 cases I drafted**. Replace and extend it with requirements you label yourself (aim for 25+); otherwise the score only measures the author's assumptions. Run `python -m eval.run_eval --mode llm` to compare Claude against the heuristic baseline. Baseline on the starter set: precision 0.69, recall 1.00 (the checklist over-flags UI-only features).

## Limitations
- Heuristic mode cannot understand meaning; use `llm` mode for real analysis.
- Tests are generated, not executed. Running them against real code is a roadmap item.
- Single-user SQLite, no authentication.

## Roadmap
1. Execute generated tests in a sandbox and record real pass/fail.
2. GitHub issue export and PR creation after approval.
3. Authentication, per-user projects, Postgres.
4. React frontend and a deployed demo.
