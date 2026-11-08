from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["actor", "acceptance", "validation", "security", "data", "performance"]


class Gap(BaseModel):
    id: str
    category: Category
    question: str
    severity: Literal["blocking", "advisory"]


class GapList(BaseModel):
    gaps: list[Gap] = []


class Requirement(BaseModel):
    id: str
    text: str


class Criterion(BaseModel):
    id: str
    requirement_id: str
    category: str
    text: str


class Spec(BaseModel):
    summary: str
    actors: list[str] = []
    requirements: list[Requirement]
    criteria: list[Criterion]
    assumptions: list[str] = []


class GeneratedTest(BaseModel):
    id: str
    requirement_id: str
    criterion_id: str
    title: str
    kind: Literal["positive", "negative", "boundary"]
    steps: list[str]
    expected: str
    executed: bool = False  # generated tests are never claimed as executed


class GeneratedTests(BaseModel):
    tests: list[GeneratedTest]


# ---- request bodies ----
class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    requirement: str = Field(min_length=10, max_length=4000)


class RequirementUpdate(BaseModel):
    requirement: str = Field(min_length=10, max_length=4000)


class AnswersIn(BaseModel):
    answers: dict[str, str]


class ApproveIn(BaseModel):
    note: str = Field(default="", max_length=500)
