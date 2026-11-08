"""Analysis backends.

HeuristicBackend: offline, deterministic, no API key. A baseline checklist, not real understanding.
AnthropicBackend: LLM-powered. Every response is validated with pydantic before use.
"""
import json
import os
import re
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from .models import Criterion, Gap, GapList, GeneratedTest, GeneratedTests, Requirement, Spec


class LLMError(Exception):
    pass


# --------------------------------------------------------------------------- heuristic
@dataclass(frozen=True)
class Check:
    category: str
    question: str
    severity: str
    pattern: re.Pattern


def _p(expr: str) -> re.Pattern:
    return re.compile(expr, re.I)


CHECKS = [
    Check("actor", "Who is the primary user or role for this feature (for example guest, registered user, admin)?", "blocking",
          _p(r"\b(users?|admins?|administrators?|customers?|guests?|managers?|developers?|patients?|students?|teachers?|employees?|visitors?|members?|operators?|owners?|roles?)\b")),
    Check("acceptance", "What observable result proves this feature works? Describe what the user sees or what the system returns.", "blocking",
          _p(r"\b(when|then|given|within|displays?|shows?|returns?|redirects?|appears?|rejects?|rejected|error message|response)\b|\b(?:is|are) (?:created|sent|saved|logged|recorded|deleted|shown|displayed)\b")),
    Check("validation", "Which inputs are valid, and what should happen when an input is invalid or missing?", "blocking",
          _p(r"\b(valid|invalid|validat\w*|format|required|requires?|reject\w*|errors?|at least|at most|maximum|minimum|max|min|length|limit\w*|expires?|confirm\w*)\b")),
    Check("security", "Who is allowed to do this, and how are access and sensitive data protected?", "blocking",
          _p(r"\b(authenticat\w*|authoriz\w*|permission\w*|roles?|encrypt\w*|hash\w*|token\w*|access control|secure|security|mfa|2fa|allowed|permit\w*|owner|restricted|logged[- ]in|signed[- ]in|registered|privacy|audit)\b")),
    Check("data", "What data is stored, and for how long? Can it be deleted?", "advisory",
          _p(r"\b(store[sd]?|saved?|database|retain\w*|delet\w*|persist\w*|records?|history|backup)\b")),
    Check("performance", "Are there speed or scale expectations (response time, number of users)?", "advisory",
          _p(r"\b(seconds?|ms|milliseconds?|latency|concurrent|scal\w*|performance|per second|at once)\b")),
]

_TEST_KIND = {"acceptance": "positive", "actor": "positive", "data": "positive",
              "validation": "negative", "security": "negative", "performance": "boundary"}


class HeuristicBackend:
    name = "heuristic"

    def find_gaps(self, requirement: str, answers: dict[str, str]) -> list[Gap]:
        return [
            Gap(id=c.category, category=c.category, question=c.question, severity=c.severity)
            for c in CHECKS
            if not c.pattern.search(requirement)
        ]

    def build_spec(self, title: str, requirement: str, answers: dict[str, str], gaps: list[Gap]) -> Spec:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", requirement) if len(s.strip()) > 3]
        reqs = [Requirement(id=f"FR-{i}", text=s) for i, s in enumerate(sentences, 1)] or [Requirement(id="FR-1", text=requirement.strip())]
        criteria: list[Criterion] = []
        for category, answer in answers.items():
            for line in re.split(r"[;\n]+", answer):
                line = line.strip()
                if line:
                    criteria.append(Criterion(id=f"AC-{len(criteria) + 1}", requirement_id="FR-1", category=category, text=line))
        if not criteria:  # requirement was already complete: one criterion per requirement
            criteria = [Criterion(id=f"AC-{i}", requirement_id=r.id, category="acceptance", text=f"The system satisfies: {r.text}") for i, r in enumerate(reqs, 1)]
        actors = [a.strip() for a in re.split(r"[;,\n]+", answers.get("actor", "")) if a.strip()]
        open_advisory = [g.question for g in gaps if g.severity == "advisory" and not answers.get(g.id, "").strip()]
        return Spec(
            summary=f"{title}: {len(reqs)} requirement(s), {len(criteria)} acceptance criteria.",
            actors=actors,
            requirements=reqs,
            criteria=criteria,
            assumptions=[f"Not specified: {q}" for q in open_advisory],
        )

    def build_tests(self, spec: Spec) -> GeneratedTests:
        tests = []
        for n, crit in enumerate(spec.criteria, 1):
            tests.append(GeneratedTest(
                id=f"T-{n}", requirement_id=crit.requirement_id, criterion_id=crit.id,
                title=f"Verify: {crit.text[:80]}",
                kind=_TEST_KIND.get(crit.category, "positive"),
                steps=["Set up the preconditions described by the requirement",
                       f"Perform the behaviour covered by: {crit.text}",
                       "Observe the system's response"],
                expected=crit.text,
            ))
        return GeneratedTests(tests=tests)


# --------------------------------------------------------------------------- LLM
GAP_SYSTEM = """You are a careful requirements analyst. Find what is missing from a software requirement.
Return ONLY JSON: {"gaps":[{"id":"<category>","category":"<category>","question":"<one specific question>","severity":"blocking|advisory"}]}
Categories: actor, acceptance, validation, security, data, performance. At most one gap per category; id must equal category.
Report a gap only if it is genuinely missing from the requirement AND the clarifying answers given so far.
"blocking" means a developer would have to guess. Return {"gaps":[]} if nothing is missing."""

SPEC_SYSTEM = """You turn a clarified software requirement into a spec. Use only facts from the requirement and the answers.
Return ONLY JSON: {"summary":"...","actors":["..."],"requirements":[{"id":"FR-1","text":"..."}],
"criteria":[{"id":"AC-1","requirement_id":"FR-1","category":"acceptance|validation|security|data|performance|actor","text":"..."}],"assumptions":["..."]}
Every criterion must be testable and reference an existing requirement id. Put anything uncertain under assumptions; never invent facts."""

TEST_SYSTEM = """You write test cases for an approved spec.
Return ONLY JSON: {"tests":[{"id":"T-1","requirement_id":"FR-1","criterion_id":"AC-1","title":"...","kind":"positive|negative|boundary","steps":["..."],"expected":"..."}]}
Every criterion needs at least one test. Include negative tests for validation and security criteria. Reference only existing ids."""


class AnthropicBackend:
    name = "llm"

    def __init__(self) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set. Set it, or use SPECGUARD_MODE=heuristic.")
        self.model = os.environ.get("SPECGUARD_MODEL", "claude-sonnet-5-5")

    def _call(self, system: str, user: str, schema: type[BaseModel]):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The anthropic package is not installed.") from exc
        client = anthropic.Anthropic()
        prompt, last_error = user, ""
        for _ in range(2):  # one retry with the validation error fed back
            try:
                msg = client.messages.create(model=self.model, max_tokens=3000, system=system,
                                             messages=[{"role": "user", "content": prompt}])
            except anthropic.APIError as exc:
                raise LLMError(f"Claude API error: {exc}") from exc
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            try:
                return schema.model_validate(json.loads(text[text.index("{"): text.rindex("}") + 1]))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)[:500]
                prompt = f"{user}\n\nYour previous reply was invalid ({last_error}). Return only valid JSON matching the schema."
        raise LLMError(f"The model returned invalid output twice: {last_error}")

    def find_gaps(self, requirement: str, answers: dict[str, str]) -> list[Gap]:
        user = f"Requirement:\n{requirement}\n\nClarifying answers so far:\n{json.dumps(answers, indent=2)}"
        return self._call(GAP_SYSTEM, user, GapList).gaps

    def build_spec(self, title: str, requirement: str, answers: dict[str, str], gaps: list[Gap]) -> Spec:
        user = f"Title: {title}\nRequirement:\n{requirement}\n\nClarifying answers:\n{json.dumps(answers, indent=2)}"
        return self._call(SPEC_SYSTEM, user, Spec)

    def build_tests(self, spec: Spec) -> GeneratedTests:
        return self._call(TEST_SYSTEM, f"Spec:\n{spec.model_dump_json(indent=2)}", GeneratedTests)


def get_backend():
    mode = os.environ.get("SPECGUARD_MODE", "").lower()
    if mode == "heuristic":
        return HeuristicBackend()
    if mode == "llm" or os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicBackend()
    return HeuristicBackend()
