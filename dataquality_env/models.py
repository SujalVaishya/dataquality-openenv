"""
dataquality_env/models.py
Typed models — uses pydantic v2 if available, else dataclasses fallback.
"""
from __future__ import annotations

try:
    from pydantic import BaseModel, Field
    _PYDANTIC = True
except ImportError:
    _PYDANTIC = False

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(str, Enum):
    INSPECT_COLUMN = "inspect_column"
    FILL_MISSING = "fill_missing"
    DROP_DUPLICATES = "drop_duplicates"
    FIX_TYPE = "fix_type"
    FLAG_OUTLIER = "flag_outlier"
    APPLY_REGEX_FIX = "apply_regex_fix"
    DROP_ROWS = "drop_rows"
    RENAME_COLUMN = "rename_column"
    SUBMIT = "submit"


if _PYDANTIC:
    class Action(BaseModel):
        action_type: ActionType
        parameters: Dict[str, Any] = Field(default_factory=dict)
        model_config = {"use_enum_values": True}

    class IssueRecord(BaseModel):
        issue_id: str
        issue_type: str
        column: Optional[str] = None
        row_indices: List[int] = Field(default_factory=list)
        severity: str
        description: str
        resolved: bool = False

    class DataMetrics(BaseModel):
        completeness: float
        consistency: float
        validity: float
        uniqueness: float

        @property
        def overall(self) -> float:
            return (self.completeness + self.consistency + self.validity + self.uniqueness) / 4.0

    class Observation(BaseModel):
        dataset_preview: List[Dict[str, Any]]
        schema: Dict[str, str]
        issue_registry: List[IssueRecord]
        action_history: List[str]
        step_count: int
        task_description: str
        metrics: DataMetrics
        last_action_result: Optional[str] = None

    class Reward(BaseModel):
        total: float
        issue_resolution_delta: float = 0.0
        metric_improvement_delta: float = 0.0
        efficiency_bonus: float = 0.0
        penalty: float = 0.0
        done: bool = False
        info: Dict[str, Any] = Field(default_factory=dict)

    class EpisodeState(BaseModel):
        task_id: str
        step_count: int
        max_steps: int
        dataset_rows: List[Dict[str, Any]]
        original_rows: List[Dict[str, Any]]
        issue_registry: List[IssueRecord]
        action_history: List[str]
        task_description: str
        metrics: DataMetrics
        done: bool = False
        score: float = 0.0

else:
    @dataclass
    class Action:
        action_type: ActionType
        parameters: Dict[str, Any] = field(default_factory=dict)

        def model_dump(self):
            return {"action_type": self.action_type, "parameters": self.parameters}

    @dataclass
    class IssueRecord:
        issue_id: str
        issue_type: str
        severity: str
        description: str
        column: Optional[str] = None
        row_indices: List[int] = field(default_factory=list)
        resolved: bool = False

        def model_dump(self):
            return asdict(self)

    @dataclass
    class DataMetrics:
        completeness: float
        consistency: float
        validity: float
        uniqueness: float

        @property
        def overall(self) -> float:
            return (self.completeness + self.consistency + self.validity + self.uniqueness) / 4.0

        def model_dump(self):
            return asdict(self)

    @dataclass
    class Observation:
        dataset_preview: List[Dict[str, Any]]
        schema: Dict[str, str]
        issue_registry: List[IssueRecord]
        action_history: List[str]
        step_count: int
        task_description: str
        metrics: DataMetrics
        last_action_result: Optional[str] = None

        def model_dump(self):
            return {
                "dataset_preview": self.dataset_preview,
                "schema": self.schema,
                "issue_registry": [i.model_dump() for i in self.issue_registry],
                "action_history": self.action_history,
                "step_count": self.step_count,
                "task_description": self.task_description,
                "metrics": self.metrics.model_dump(),
                "last_action_result": self.last_action_result,
            }

    @dataclass
    class Reward:
        total: float
        issue_resolution_delta: float = 0.0
        metric_improvement_delta: float = 0.0
        efficiency_bonus: float = 0.0
        penalty: float = 0.0
        done: bool = False
        info: Dict[str, Any] = field(default_factory=dict)

        def model_dump(self):
            return asdict(self)

    @dataclass
    class EpisodeState:
        task_id: str
        step_count: int
        max_steps: int
        dataset_rows: List[Dict[str, Any]]
        original_rows: List[Dict[str, Any]]
        issue_registry: List[IssueRecord]
        action_history: List[str]
        task_description: str
        metrics: DataMetrics
        done: bool = False
        score: float = 0.0

        def model_dump(self):
            return {
                "task_id": self.task_id,
                "step_count": self.step_count,
                "max_steps": self.max_steps,
                "dataset_rows": self.dataset_rows,
                "original_rows": self.original_rows,
                "issue_registry": [i.model_dump() for i in self.issue_registry],
                "action_history": self.action_history,
                "task_description": self.task_description,
                "metrics": self.metrics.model_dump(),
                "done": self.done,
                "score": self.score,
            }
