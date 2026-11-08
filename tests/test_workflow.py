import pytest

from app.workflow import State, WorkflowError, allowed_actions, require, state_after_analysis


def test_cannot_approve_before_spec_ready():
    with pytest.raises(WorkflowError):
        require("approve", State.NEEDS_CLARIFICATION)


def test_cannot_generate_tests_before_approval():
    for s in (State.SUBMITTED, State.NEEDS_CLARIFICATION, State.SPEC_READY):
        with pytest.raises(WorkflowError):
            require("generate_tests", s)


def test_allowed_actions_by_state():
    assert allowed_actions(State.SUBMITTED) == ["revise", "analyze"]
    assert "approve" in allowed_actions(State.SPEC_READY)
    assert allowed_actions(State.TESTS_READY) == ["revise"]


def test_open_blocking_gaps_keep_project_in_clarification():
    assert state_after_analysis(2) is State.NEEDS_CLARIFICATION
    assert state_after_analysis(0) is State.SPEC_READY
