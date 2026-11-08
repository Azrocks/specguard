"""Analysis backends.

HeuristicBackend: offline, deterministic, no API key. A baseline checklist, not real understanding.
AnthropicBackend / OpenAICompatBackend: LLM-powered. Every response is validated with pydantic before use.
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
GAP_SYSTEM = """You are a strict requirements reviewer. Find what is missing from a software requirement.
Return ONLY JSON: {"gaps":[{"id":"<category>","category":"<category>","question":"<one specific question>","severity":"blocking|advisory"}]}
Categories (one question each, id must equal category):
- actor: who uses the feature or which roles exist
- acceptance: the observable result that proves it works
- validation: valid inputs and what happens on invalid or missing input
- security: who may do it and how access and sensitive data are protected
- data: what is stored and for how long
- performance: speed or scale expectations
A category counts as covered ONLY if the requirement or the clarifying answers explicitly state it. Never assume sensible defaults and never treat a topic as covered because it is implied by the feature name.
Mark actor, acceptance, validation and security "blocking" when missing, data and performance "advisory". Short, vague requirements usually have several blocking gaps.
Ask one focused question per category; do not combine topics.
Example: for "Create a login page." the gaps are actor, acceptance, validation and security (all blocking), plus data and performance (advisory).
Return {"gaps":[]} only if every category is explicitly covered."""

SPEC_SYSTEM = """You turn a clarified software requirement into a spec. Use only facts from the requirement and the answers.
Return ONLY JSON: {"summary":"...","actors":["..."],"requirements":[{"id":"FR-1","text":"..."}],
"criteria":[{"id":"AC-1","requirement_id":"FR-1","category":"acceptance|validation|security|data|performance|actor","text":"..."}],"assumptions":["..."]}
Every criterion must be testable and reference an existing requirement id. Put anything uncertain under assumptions; never invent facts."""

TEST_SYSTEM = """You write test cases for an approved spec.
Return ONLY JSON: {"tests":[{"id":"T-1","requirement_id":"FR-1","criterion_id":"AC-1","title":"...","kind":"positive|negative|boundary","steps":["..."],"expected":"..."}]}
Every criterion needs at least one test. Include negative tests for validation and security criteria. Reference only existing ids."""

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"  # change with SPECGUARD_MODEL if Google renames it


class _LLMBackend:
    """Shared prompt, validation and retry logic. Subclasses only implement _complete()."""

    name = "llm"
    label = "LLM"

    def _complete(self, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def _call(self, system: str, user: str, schema: type[BaseModel]):
        prompt, last_error = user, ""
        for _ in range(2):  # one retry with the validation error fed back
            text = self._complete(system, prompt)
            try:
                return schema.model_validate(json.loads(text[text.index("{"): text.rindex("}") + 1]))
            except ValueError as exc:  # JSON errors, missing braces and pydantic errors
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


class AnthropicBackend(_LLMBackend):
    def __init__(self) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        self.model = os.environ.get("SPECGUARD_MODEL", "claude-sonnet-5-5")
        self.label = "Claude"

    def _complete(self, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The anthropic package is not installed. Run: pip install anthropic") from exc
        try:
            msg = anthropic.Anthropic().messages.create(
                model=self.model, max_tokens=3000, system=system, messages=[{"role": "user", "content": user}])
        except anthropic.APIError as exc:
            raise LLMError(f"Claude API error: {exc}") from exc
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAICompatBackend(_LLMBackend):
    """Any OpenAI-compatible chat endpoint: Google Gemini's free tier, Groq, Ollama, and others."""

    def __init__(self, base_url: str, api_key: str, model: str, label: str) -> None:
        self.base_url, self.api_key, self.model, self.label = base_url.rstrip("/"), api_key, model, label

    def _complete(self, system: str, user: str) -> str:
        import time

        import httpx

        payload = {"model": self.model, "temperature": 0,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(4):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=90)
            except httpx.HTTPError as exc:
                raise LLMError(f"Could not reach {self.label}: {exc}") from exc
            if r.status_code in (429, 503) and attempt < 3:  # rate limit or temporary overload: back off and retry
                time.sleep(4 * (attempt + 1))
                continue
            break
        if r.status_code == 503:
            raise LLMError(f"{self.label} is overloaded right now. Wait a minute and try again.")
        if r.status_code == 429:
            raise LLMError(f"{self.label} rate limit reached. Wait a minute and try again.")
        if r.status_code in (401, 403):
            raise LLMError(f"{self.label} rejected the API key (HTTP {r.status_code}). Check the key.")
        if r.status_code == 404:
            raise LLMError(f"{self.label}: model or endpoint not found. Check SPECGUARD_MODEL ({self.model}).")
        if r.status_code >= 400:
            raise LLMError(f"{self.label} error {r.status_code}: {r.text[:300]}")
        try:
            return r.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected response format from {self.label}") from exc


def _llm_backend():
    provider = os.environ.get("SPECGUARD_PROVIDER", "").lower()
    if not provider:  # auto-detect from whichever key is present
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.environ.get("SPECGUARD_BASE_URL"):
            provider = "custom"
    model = os.environ.get("SPECGUARD_MODEL", "")
    if provider == "anthropic":
        return AnthropicBackend()
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set.")
        return OpenAICompatBackend(GEMINI_BASE_URL, key, model or GEMINI_DEFAULT_MODEL, "Gemini (free tier)")
    if provider == "custom":
        base = os.environ.get("SPECGUARD_BASE_URL", "")
        if not base or not model:
            raise LLMError("Custom provider needs SPECGUARD_BASE_URL and SPECGUARD_MODEL (SPECGUARD_API_KEY if required).")
        return OpenAICompatBackend(base, os.environ.get("SPECGUARD_API_KEY", "none"), model, model)
    raise LLMError("No LLM provider configured. Set GEMINI_API_KEY (free), ANTHROPIC_API_KEY, or SPECGUARD_BASE_URL.")


def get_backend():
    mode = os.environ.get("SPECGUARD_MODE", "").lower()
    if mode == "heuristic":
        return HeuristicBackend()
    has_llm = any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "SPECGUARD_BASE_URL"))
    if mode == "llm" or has_llm:
        return _llm_backend()
    return HeuristicBackend()