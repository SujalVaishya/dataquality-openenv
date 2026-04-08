"""
dataquality_env/env.py
DataQualityEnv — full OpenEnv-compliant environment.
Implements step(), reset(), state() with typed Pydantic models.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

from dataquality_env.models import (
    Action, ActionType, DataMetrics, EpisodeState,
    IssueRecord, Observation, Reward,
)
from dataquality_env.tasks import get_task

# Valid category maps for hard task normalization
_STATUS_MAP = {
    "scheduled": "scheduled", "Scheduled": "scheduled", "SCHEDULED": "scheduled",
    "completed": "completed", "Completed": "completed", "COMPLETED": "completed",
    "cancelled": "cancelled", "canceled": "cancelled", "cancled": "cancelled", "CANCELLED": "cancelled",
    "no_show": "no_show", "no show": "no_show", "no-show": "no_show", "NO_SHOW": "no_show",
}
_DEPT_MAP = {
    "cardiology": "cardiology", "Cardiology": "cardiology", "CARDIOLOGY": "cardiology",
    "orthopedics": "orthopedics", "Orthopedics": "orthopedics", "ortho": "orthopedics", "ORTHO": "orthopedics",
    "neurology": "neurology", "Neurology": "neurology", "neuro": "neurology", "NEURO": "neurology",
    "pediatrics": "pediatrics", "Pediatrics": "pediatrics", "kids": "pediatrics",
    "oncology": "oncology", "Oncology": "oncology",
}


class DataQualityEnv:
    """
    OpenEnv-compliant environment for data quality triage.

    Agents interact with a messy tabular dataset by issuing structured actions
    to identify and resolve data quality issues.

    Usage:
        env = DataQualityEnv(task_id="task_easy")
        obs = env.reset()
        while not done:
            action = agent.decide(obs)
            obs, reward, done, info = env.step(action)
        final_score = env.state().score
    """

    TASK_IDS = ["task_easy", "task_medium", "task_hard"]

    def __init__(self, task_id: str = "task_easy", seed: int = 42):
        if task_id not in self.TASK_IDS:
            raise ValueError(f"task_id must be one of {self.TASK_IDS}")
        self.task_id = task_id
        self.seed = seed
        self._state: Optional[EpisodeState] = None
        self._grader = None

    # ─────────────────────────────────────────────
    # OpenEnv core API
    # ─────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment to a fresh episode. Returns initial observation."""
        rows, issues, description, grader, max_steps = get_task(self.task_id, self.seed)
        self._grader = grader

        initial_metrics = self._compute_metrics(rows, issues)
        self._state = EpisodeState(
            task_id=self.task_id,
            step_count=0,
            max_steps=max_steps,
            dataset_rows=copy.deepcopy(rows),
            original_rows=copy.deepcopy(rows),
            issue_registry=copy.deepcopy(issues),
            action_history=[],
            task_description=description,
            metrics=initial_metrics,
            done=False,
            score=0.0,
        )
        return self._make_observation(last_result="Episode started. Inspect the dataset and resolve issues.")

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one action. Returns (observation, reward, done, info).
        Reward is a Reward pydantic model; done is bool; info is dict.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        prev_metrics = copy.deepcopy(self._state.metrics)
        prev_resolved = sum(1 for iss in self._state.issue_registry if iss.resolved)

        result_msg, penalty = self._execute_action(action)

        self._state.step_count += 1
        self._state.action_history.append(
            f"[{self._state.step_count}] {action.action_type}: {result_msg}"
        )

        # Refresh metrics and issue registry
        self._state.metrics = self._compute_metrics(
            self._state.dataset_rows, self._state.issue_registry
        )
        self._refresh_issue_resolved_state()

        # Compute reward components
        metric_delta = self._state.metrics.overall - prev_metrics.overall
        new_resolved = sum(1 for iss in self._state.issue_registry if iss.resolved)
        resolution_delta = (new_resolved - prev_resolved) / max(len(self._state.issue_registry), 1)

        # Efficiency bonus for solving issues quickly
        steps_remaining_ratio = 1.0 - (self._state.step_count / self._state.max_steps)
        efficiency_bonus = 0.02 * steps_remaining_ratio if resolution_delta > 0 else 0.0

        # Check done conditions
        done = False
        if action.action_type == ActionType.SUBMIT:
            done = True
        elif self._state.step_count >= self._state.max_steps:
            done = True
            penalty += -0.05  # timeout penalty

        self._state.done = done

        # Compute final score on done
        if done:
            self._state.score = self._grader(
                self._state.dataset_rows,
                self._state.original_rows,
                self._state.action_history,
            )

        total_reward = round(
            0.5 * metric_delta
            + 0.3 * resolution_delta
            + efficiency_bonus
            + penalty,
            4,
        )
        # Clip reward to [-1, 1]
        total_reward = max(-1.0, min(1.0, total_reward))

        reward = Reward(
            total=total_reward,
            issue_resolution_delta=resolution_delta,
            metric_improvement_delta=metric_delta,
            efficiency_bonus=efficiency_bonus,
            penalty=penalty,
            done=done,
            info={"step": self._state.step_count, "score": self._state.score if done else None},
        )

        obs = self._make_observation(last_result=result_msg)
        return obs, reward, done, reward.info

    def state(self) -> EpisodeState:
        """Return full internal state."""
        if self._state is None:
            raise RuntimeError("Call reset() before state()")
        return copy.deepcopy(self._state)

    # ─────────────────────────────────────────────
    # Action execution
    # ─────────────────────────────────────────────

    def _execute_action(self, action: Action) -> Tuple[str, float]:
        """Execute action, return (result_message, penalty)."""
        rows = self._state.dataset_rows
        p = action.parameters
        atype = action.action_type

        try:
            if atype == ActionType.INSPECT_COLUMN:
                return self._action_inspect_column(rows, p), 0.0

            elif atype == ActionType.FILL_MISSING:
                return self._action_fill_missing(rows, p), 0.0

            elif atype == ActionType.DROP_DUPLICATES:
                return self._action_drop_duplicates(rows, p), 0.0

            elif atype == ActionType.FIX_TYPE:
                return self._action_fix_type(rows, p), 0.0

            elif atype == ActionType.FLAG_OUTLIER:
                return self._action_flag_outlier(rows, p), 0.0

            elif atype == ActionType.APPLY_REGEX_FIX:
                return self._action_apply_regex_fix(rows, p), 0.0

            elif atype == ActionType.DROP_ROWS:
                return self._action_drop_rows(rows, p), 0.0

            elif atype == ActionType.RENAME_COLUMN:
                return self._action_rename_column(rows, p), 0.0

            elif atype == ActionType.SUBMIT:
                score = self._grader(rows, self._state.original_rows, self._state.action_history)
                return f"Submitted. Final score: {score:.4f}", 0.0

            else:
                return f"Unknown action type: {atype}", -0.05

        except Exception as e:
            return f"Action failed: {str(e)}", -0.02

    def _action_inspect_column(self, rows: List[Dict], p: Dict) -> str:
        col = p.get("column")
        if not col:
            # List all columns
            if not rows:
                return "Dataset is empty."
            cols = list(rows[0].keys())
            return f"Columns: {cols}. Total rows: {len(rows)}"

        values = [r.get(col) for r in rows]
        non_null = [v for v in values if v is not None]
        null_count = len(values) - len(non_null)
        unique_vals = list(set(str(v) for v in non_null))[:10]
        types_seen = list(set(type(v).__name__ for v in non_null))
        return (
            f"Column '{col}': {len(values)} rows, {null_count} nulls, "
            f"types={types_seen}, sample_uniques={unique_vals[:5]}"
        )

    def _action_fill_missing(self, rows: List[Dict], p: Dict) -> str:
        col = p.get("column")
        fill_value = p.get("fill_value")
        strategy = p.get("strategy", "constant")  # constant | forward_fill | mean

        if not col:
            return "Error: 'column' parameter required."
        if col not in (rows[0].keys() if rows else []):
            return f"Error: column '{col}' not found."

        count = 0
        if strategy == "constant":
            if fill_value is None:
                return "Error: 'fill_value' required for strategy=constant"
            for r in rows:
                if r.get(col) is None:
                    r[col] = fill_value
                    count += 1
        elif strategy == "forward_fill":
            last_val = None
            for r in rows:
                if r.get(col) is not None:
                    last_val = r[col]
                elif last_val is not None:
                    r[col] = last_val
                    count += 1
        elif strategy == "mean":
            nums = [r[col] for r in rows if isinstance(r.get(col), (int, float))]
            if not nums:
                return f"No numeric values found in '{col}' for mean strategy."
            mean_val = round(sum(nums) / len(nums), 2)
            for r in rows:
                if r.get(col) is None:
                    r[col] = mean_val
                    count += 1
        else:
            return f"Unknown strategy '{strategy}'. Use: constant, forward_fill, mean"

        return f"Filled {count} missing values in '{col}' using strategy='{strategy}'."

    def _action_drop_duplicates(self, rows: List[Dict], p: Dict) -> str:
        subset = p.get("subset")  # list of columns to check, or None for all
        keep = p.get("keep", "first")  # first | last

        before = len(rows)
        seen: set = set()
        to_keep = []
        indices_to_check = list(range(len(rows))) if keep == "first" else list(reversed(range(len(rows))))

        kept_indices = set()
        for idx in indices_to_check:
            r = rows[idx]
            if subset:
                key = tuple(r.get(c) for c in subset)
            else:
                key = tuple(sorted(r.items()))
            if key not in seen:
                seen.add(key)
                kept_indices.add(idx)

        new_rows = [rows[i] for i in sorted(kept_indices)]
        removed = before - len(new_rows)
        self._state.dataset_rows[:] = new_rows
        return f"Dropped {removed} duplicate rows. Remaining: {len(new_rows)} rows."

    def _action_fix_type(self, rows: List[Dict], p: Dict) -> str:
        col = p.get("column")
        target_type = p.get("target_type")  # float | int | str | date
        strip_chars = p.get("strip_chars", "")  # e.g. "$,€"

        if not col or not target_type:
            return "Error: 'column' and 'target_type' required."

        count = 0
        errors = 0
        iso_date = re.compile(r"^\d{4}-\d{2}-\d{2}$")

        for r in rows:
            val = r.get(col)
            if val is None:
                continue
            try:
                if target_type == "float":
                    cleaned = str(val)
                    for ch in strip_chars:
                        cleaned = cleaned.replace(ch, "")
                    cleaned = cleaned.replace(",", "")
                    r[col] = float(cleaned)
                    count += 1
                elif target_type == "int":
                    cleaned = str(val)
                    for ch in strip_chars:
                        cleaned = cleaned.replace(ch, "")
                    r[col] = int(float(cleaned))
                    count += 1
                elif target_type == "str":
                    r[col] = str(val)
                    count += 1
                elif target_type == "date":
                    # Attempt common date parsing → normalize to ISO YYYY-MM-DD
                    normalized = self._normalize_date(str(val))
                    if normalized:
                        r[col] = normalized
                        count += 1
                    else:
                        errors += 1
                elif target_type == "datetime":
                    normalized = self._normalize_datetime(str(val))
                    if normalized:
                        r[col] = normalized
                        count += 1
                    else:
                        errors += 1
            except (ValueError, TypeError):
                errors += 1

        return f"Fixed type for '{col}' → {target_type}: {count} converted, {errors} failed."

    def _normalize_date(self, val: str) -> Optional[str]:
        """Normalize common date formats to YYYY-MM-DD."""
        patterns = [
            (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
            (re.compile(r"^(\d{2})/(\d{2})/(\d{4})$"), lambda m: f"{m.group(3)}-{m.group(1)}-{m.group(2)}"),
            (re.compile(r"^(\d{2})-(\d{2})-(\d{4})$"), lambda m: f"{m.group(3)}-{m.group(1)}-{m.group(2)}"),
            (re.compile(r"^(\d{4})/(\d{2})/(\d{2})$"), lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
            (re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$"), lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}"),
            (re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$"), lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}"),
            (re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$"), lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),
            # Jan 15 2024
            (re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\s+(\d{4})$", re.I),
             lambda m: f"{m.group(3)}-{self._month_num(m.group(1)):02d}-{int(m.group(2)):02d}"),
        ]
        for pattern, formatter in patterns:
            match = pattern.match(val.strip())
            if match:
                try:
                    return formatter(match)
                except Exception:
                    continue
        return None

    def _normalize_datetime(self, val: str) -> Optional[str]:
        """Normalize common datetime formats to YYYY-MM-DDThh:mm:ss."""
        patterns = [
            (re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})$"),
             lambda m: val.strip()),
            (re.compile(r"^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})$"),
             lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:00"),
            (re.compile(r"^(\d{2})/(\d{2})/(\d{4})\s+(\d{1,2}):(\d{2})\s*(AM|PM)$", re.I),
             lambda m: self._parse_12h(m)),
            # March 1 2024
            (re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})\s+(\d{4})$", re.I),
             lambda m: f"{m.group(3)}-{self._month_num(m.group(1)):02d}-{int(m.group(2)):02d}T00:00:00"),
        ]
        for pattern, formatter in patterns:
            match = pattern.match(val.strip())
            if match:
                try:
                    return formatter(match)
                except Exception:
                    continue
        return None

    def _month_num(self, name: str) -> int:
        months = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                  "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
                  "january":1,"february":2,"march":3,"april":4,"june":6,
                  "july":7,"august":8,"september":9,"october":10,"november":11,"december":12}
        return months.get(name.lower()[:3], 1)

    def _parse_12h(self, m) -> str:
        h = int(m.group(4))
        ampm = m.group(6).upper()
        if ampm == "PM" and h != 12:
            h += 12
        elif ampm == "AM" and h == 12:
            h = 0
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}T{h:02d}:{m.group(5)}:00"

    def _action_flag_outlier(self, rows: List[Dict], p: Dict) -> str:
        col = p.get("column")
        min_val = p.get("min_val")
        max_val = p.get("max_val")
        action_on_outlier = p.get("action", "flag")  # flag | drop | clip

        if not col:
            return "Error: 'column' required."

        flagged = []
        for i, r in enumerate(rows):
            val = r.get(col)
            if not isinstance(val, (int, float)):
                continue
            is_outlier = False
            if min_val is not None and val < min_val:
                is_outlier = True
            if max_val is not None and val > max_val:
                is_outlier = True
            if is_outlier:
                flagged.append(i)
                if action_on_outlier == "clip":
                    if min_val is not None and val < min_val:
                        r[col] = min_val
                    if max_val is not None and val > max_val:
                        r[col] = max_val

        if action_on_outlier == "drop" and flagged:
            keep_indices = set(range(len(rows))) - set(flagged)
            new_rows = [rows[i] for i in sorted(keep_indices)]
            self._state.dataset_rows[:] = new_rows
            return f"Dropped {len(flagged)} outlier rows in '{col}'."

        return f"Flagged {len(flagged)} outliers in '{col}' (action={action_on_outlier})."

    def _action_apply_regex_fix(self, rows: List[Dict], p: Dict) -> str:
        col = p.get("column")
        pattern = p.get("pattern")
        replacement = p.get("replacement", "")
        normalize_map = p.get("normalize_map")  # dict: {"bad_val": "good_val", ...}

        if not col:
            return "Error: 'column' required."

        count = 0
        if normalize_map:
            for r in rows:
                val = r.get(col)
                if val in normalize_map:
                    r[col] = normalize_map[val]
                    count += 1
            return f"Applied normalize_map to '{col}': {count} values changed."

        if not pattern:
            return "Error: 'pattern' or 'normalize_map' required."

        compiled = re.compile(pattern)
        for r in rows:
            val = r.get(col)
            if val is not None:
                new_val = compiled.sub(replacement, str(val))
                if new_val != str(val):
                    r[col] = new_val
                    count += 1
        return f"Applied regex pattern to '{col}': {count} values changed."

    def _action_drop_rows(self, rows: List[Dict], p: Dict) -> str:
        condition_col = p.get("column")
        condition = p.get("condition")  # "is_null" | "not_in" | "less_than" | "greater_than"
        value = p.get("value")
        row_indices = p.get("row_indices")  # explicit list of indices

        before = len(rows)

        if row_indices is not None:
            drop_set = set(row_indices)
            new_rows = [r for i, r in enumerate(rows) if i not in drop_set]
        elif condition == "is_null":
            new_rows = [r for r in rows if r.get(condition_col) is not None]
        elif condition == "not_in" and value is not None:
            valid_set = set(value) if isinstance(value, list) else {value}
            new_rows = [r for r in rows if r.get(condition_col) in valid_set]
        elif condition == "less_than" and value is not None:
            new_rows = [r for r in rows if not (isinstance(r.get(condition_col), (int, float)) and r[condition_col] < value)]
        elif condition == "greater_than" and value is not None:
            new_rows = [r for r in rows if not (isinstance(r.get(condition_col), (int, float)) and r[condition_col] > value)]
        else:
            return "Error: Provide 'row_indices' or valid condition+column+value."

        removed = before - len(new_rows)
        self._state.dataset_rows[:] = new_rows
        return f"Dropped {removed} rows. Remaining: {len(new_rows)} rows."

    def _action_rename_column(self, rows: List[Dict], p: Dict) -> str:
        old_name = p.get("old_name")
        new_name = p.get("new_name")
        if not old_name or not new_name:
            return "Error: 'old_name' and 'new_name' required."
        count = 0
        for r in rows:
            if old_name in r:
                r[new_name] = r.pop(old_name)
                count += 1
        return f"Renamed column '{old_name}' → '{new_name}' in {count} rows."

    # ─────────────────────────────────────────────
    # Metrics & helpers
    # ─────────────────────────────────────────────

    def _compute_metrics(self, rows: List[Dict], issues: List[IssueRecord]) -> DataMetrics:
        if not rows:
            return DataMetrics(completeness=0.0, consistency=0.0, validity=0.0, uniqueness=0.0)

        n = len(rows)
        cols = list(rows[0].keys())

        # Completeness: fraction of non-null cells
        total_cells = n * len(cols)
        non_null = sum(1 for r in rows for v in r.values() if v is not None)
        completeness = non_null / total_cells if total_cells > 0 else 1.0

        # Uniqueness: no duplicate rows (by all cols)
        seen: set = set()
        uniq = 0
        for r in rows:
            key = tuple(sorted(r.items()))
            if key not in seen:
                seen.add(key)
                uniq += 1
        uniqueness = uniq / n

        # Validity: fraction of unresolved critical/major issues relative to row count
        unresolved_critical = sum(1 for iss in issues
                                  if not iss.resolved and iss.severity in ("critical", "major"))
        validity = max(0.0, 1.0 - (unresolved_critical * 0.1))

        # Consistency: based on unresolved format/type issues
        format_issues = sum(1 for iss in issues
                            if not iss.resolved and iss.issue_type in ("format", "type_error"))
        consistency = max(0.0, 1.0 - (format_issues * 0.15))

        return DataMetrics(
            completeness=round(min(1.0, completeness), 4),
            consistency=round(min(1.0, consistency), 4),
            validity=round(min(1.0, validity), 4),
            uniqueness=round(min(1.0, uniqueness), 4),
        )

    def _refresh_issue_resolved_state(self):
        """Re-evaluate which issues are still present after actions."""
        rows = self._state.dataset_rows
        for iss in self._state.issue_registry:
            if iss.resolved:
                continue
            if iss.issue_type == "missing":
                col = iss.column
                remaining = [i for i, r in enumerate(rows) if r.get(col) is None]
                if not remaining:
                    iss.resolved = True
                    iss.row_indices = []
                else:
                    iss.row_indices = remaining
            elif iss.issue_type == "duplicate":
                seen: set = set()
                dups = []
                for i, r in enumerate(rows):
                    if iss.column:
                        key = r.get(iss.column)
                    else:
                        key = tuple(sorted(r.items()))
                    if key in seen:
                        dups.append(i)
                    else:
                        seen.add(key)
                if not dups:
                    iss.resolved = True
                    iss.row_indices = []
                else:
                    iss.row_indices = dups
            elif iss.issue_type == "type_error":
                col = iss.column
                bad = [i for i, r in enumerate(rows) if isinstance(r.get(col), str)
                       and any(c in str(r.get(col, "")) for c in ["$", "€", "£"])]
                if not bad:
                    iss.resolved = True
                else:
                    iss.row_indices = bad
            elif iss.issue_type in ("format", "outlier"):
                # Re-check via grader metrics — mark as resolved if no remaining bad rows
                col = iss.column
                if col:
                    remaining = self._count_format_issues(rows, iss)
                    if remaining == 0:
                        iss.resolved = True
                        iss.row_indices = []

    def _count_format_issues(self, rows: List[Dict], iss: IssueRecord) -> int:
        import re as _re
        col = iss.column
        if iss.issue_type == "outlier" and col == "age":
            return sum(1 for r in rows if isinstance(r.get(col), int) and (r[col] < 0 or r[col] > 130))
        if iss.issue_type == "outlier" and col == "quantity":
            return sum(1 for r in rows if isinstance(r.get(col), (int, float)) and r[col] < 0)
        if iss.issue_type == "format" and col in ("appointment_date", "date"):
            iso = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
            return sum(1 for r in rows if r.get(col) and not iso.match(str(r[col])))
        if iss.issue_type == "format" and col == "created_timestamp":
            iso = _re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
            return sum(1 for r in rows if r.get(col) and not iso.match(str(r[col])))
        if iss.issue_type == "format" and col == "status":
            valid = {"scheduled", "completed", "cancelled", "no_show"}
            return sum(1 for r in rows if r.get(col) not in valid)
        if iss.issue_type == "format" and col == "department":
            valid = {"cardiology", "orthopedics", "neurology", "pediatrics", "oncology"}
            return sum(1 for r in rows if r.get(col) not in valid)
        return len(iss.row_indices)

    def _make_observation(self, last_result: str = "") -> Observation:
        s = self._state
        preview = s.dataset_rows[:20]
        schema = {}
        if s.dataset_rows:
            for col, val in s.dataset_rows[0].items():
                schema[col] = type(val).__name__ if val is not None else "unknown"
        return Observation(
            dataset_preview=preview,
            schema=schema,
            issue_registry=s.issue_registry,
            action_history=s.action_history[-10:],  # last 10 actions
            step_count=s.step_count,
            task_description=s.task_description,
            metrics=s.metrics,
            last_action_result=last_result,
        )
