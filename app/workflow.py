"""The workflow rules live here, in code, not in prompts.

SUBMITTED -> NEEDS_CLARIFICATION -> SPEC_READY -> APPROVED -> TESTS_READY
An LLM can propose, but it can never skip a stage.
"""
from enum import Enum


class State(str, Enum):
    SUBMITTED = "SUBMITTED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    SPEC_READY = "SPEC_READY"
    APPROVED = "APPROVED"
    TESTS_READY = "TESTS_READY"


class WorkflowError(Exception):
    pass


ALLOWED: dict[str, set[State]] = {
    "revise": set(State),
    "analyze": {State.SUBMITTED, State.NEEDS_CLARIFICATION},
    "answer": {State.NEEDS_CLARIFICATION},
    "approve": {State.SPEC_READY},
    "generate_tests": {State.APPROVED},
}

HINTS = {
    "analyze": "Analysis only runs on a new or clarifying project.",
    "answer": "Answers can only be added while the project needs clarification.",
    "approve": "Approval needs a ready spec: answer every blocking question and re-run analysis first.",
    "generate_tests": "Tests are generated only from an approved spec.",
}


def allowed_actions(state: State) -> list[str]:
    return [action for action, states in ALLOWED.items() if state in states]


def require(action: str, state: State) -> None:
    if state not in ALLOWED[action]:
        raise WorkflowError(f"Can't {action.replace('_', ' ')} while the project is {state.value}. {HINTS.get(action, '')}".strip())


def state_after_analysis(open_blocking: int) -> State:
    return State.NEEDS_CLARIFICATION if open_blocking else State.SPEC_READY
