---
title: DataQuality OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - data-engineering
  - data-quality
  - reinforcement-learning
  - agent-evaluation
---

# 🧹 DataQuality OpenEnv

**A real-world OpenEnv environment for training and evaluating AI agents on data quality triage.**

Data engineers spend a significant portion of their time identifying and fixing data quality issues — missing values, type errors, duplicate records, format violations, outliers, and referential integrity problems. This environment simulates that work as a structured, episodic agent task with partial-progress rewards and deterministic graders.

---

## Why This Environment?

Data quality triage is a genuinely hard, high-value real-world task:

- It requires **inspection** (understanding what's wrong), **planning** (deciding what to fix first), and **execution** (applying the right operations in the right order)
- It has **rich partial progress signals** — each fix measurably improves the dataset
- It scales in difficulty: a simple contacts CSV vs. a production healthcare dataset
- It maps cleanly to agent capabilities: tool use, state tracking, multi-step reasoning

This fills a gap in the OpenEnv ecosystem — no existing environment models data engineering tasks.

---

## Action Space

Agents issue JSON actions with an `action_type` and `parameters` dict:

| Action Type | Parameters | Description |
|---|---|---|
| `inspect_column` | `column` (optional) | Get dtype, null count, unique samples for a column. No `column` → list all columns. |
| `fill_missing` | `column`, `fill_value`, `strategy` (constant/forward_fill/mean) | Fill null values in a column |
| `drop_duplicates` | `subset` (list of cols or null), `keep` (first/last) | Remove duplicate rows |
| `fix_type` | `column`, `target_type` (float/int/str/date/datetime), `strip_chars` | Convert column dtype; strips currency symbols etc. |
| `flag_outlier` | `column`, `min_val`, `max_val`, `action` (drop/clip/flag) | Handle out-of-range numeric values |
| `apply_regex_fix` | `column`, `normalize_map` OR `pattern`+`replacement` | Normalize categorical values or apply regex substitution |
| `drop_rows` | `column`, `condition` (is_null/not_in/less_than/greater_than), `value` OR `row_indices` | Remove rows matching a condition |
| `rename_column` | `old_name`, `new_name` | Rename a column |
| `submit` | _(none)_ | End the episode and trigger final scoring |

**Example action:**
```json
{
  "action_type": "fix_type",
  "parameters": {
    "column": "revenue",
    "target_type": "float",
    "strip_chars": "$,"
  }
}
```

---

## Observation Space

Each step returns an `Observation` with:

| Field | Type | Description |
|---|---|---|
| `dataset_preview` | `List[Dict]` | First 20 rows of the current dataset |
| `schema` | `Dict[str, str]` | Column names → inferred Python type |
| `issue_registry` | `List[IssueRecord]` | All detected issues with `resolved` flag |
| `action_history` | `List[str]` | Last 10 actions taken (with results) |
| `step_count` | `int` | Number of steps taken this episode |
| `task_description` | `str` | Natural language description of the objective |
| `metrics` | `DataMetrics` | Four quality scores: completeness, consistency, validity, uniqueness (each 0–1) |
| `last_action_result` | `str` | Human-readable result of the most recent action |

**IssueRecord fields:** `issue_id`, `issue_type` (missing/duplicate/type_error/format/outlier), `column`, `row_indices`, `severity` (critical/major/minor), `description`, `resolved`

---

## Tasks

### Task 1: Basic Completeness Fix (`task_easy`)
**Difficulty:** Easy | **Max steps:** 15 | **Target score:** ≥ 0.85

A customer contacts CSV (35 rows) has:
- ~30% of `email` values missing → fill with `unknown@example.com`
- ~25% of `phone` values missing → fill with `N/A`
- 5 exact duplicate rows → remove

**Grader weights:** email completeness 40%, phone completeness 30%, no duplicates 30%

**Optimal solution (4 actions):**
```json
[
  {"action_type": "fill_missing", "parameters": {"column": "email", "fill_value": "unknown@example.com", "strategy": "constant"}},
  {"action_type": "fill_missing", "parameters": {"column": "phone", "fill_value": "N/A", "strategy": "constant"}},
  {"action_type": "drop_duplicates", "parameters": {}},
  {"action_type": "submit", "parameters": {}}
]
```

---

### Task 2: Type Errors & Format Violations (`task_medium`)
**Difficulty:** Medium | **Max steps:** 20 | **Target score:** ≥ 0.80

A sales transactions dataset (40 rows) has:
- `revenue` stored as strings with currency symbols (`"$1,234.56"`) → convert to float
- `date` column in 5 mixed non-ISO formats (`MM/DD/YYYY`, `DD-MM-YYYY`, `Jan 15 2024`, etc.) → normalize to `YYYY-MM-DD`
- ~15% of `quantity` values are negative → drop those rows
- Duplicate `transaction_id` values → deduplicate

**Grader weights:** revenue numeric 35%, dates ISO 25%, no negative qty 20%, unique transaction IDs 20%

---

### Task 3: Multi-Issue Production Dataset (`task_hard`)
**Difficulty:** Hard | **Max steps:** 30 | **Target score:** ≥ 0.75

A healthcare appointments dataset (65 rows) with 7 simultaneous issue types:
1. **Missing `patient_id`** (~15% of rows) → drop those rows
2. **`appointment_date` format violations** (~40% non-ISO) → standardize to `YYYY-MM-DD`
3. **`created_timestamp` format violations** (~35% non-ISO) → standardize to `YYYY-MM-DDThh:mm:ss`
4. **Age outliers** (negative ages, age > 130) → drop those rows
5. **Invalid `status` values** (`"Scheduled"`, `"cancled"`, `"no show"`) → normalize to `{scheduled, completed, cancelled, no_show}`
6. **Invalid `department` values** (`"ORTHO"`, `"neuro"`, `"kids"`) → normalize to valid set
7. **Duplicate `appointment_id`** → remove duplicates

**Grader weights:** patient_id 20%, appt_date 15%, created_ts 15%, age 15%, status 15%, dept 10%, uniqueness 10%

This task challenges frontier models because all 7 issues must be identified and addressed in the right order, with some interdependencies (e.g., dropping null patient_ids first reduces the duplicate count).

---

## Reward Function

The reward at each step is a combination of:

| Component | Weight | Description |
|---|---|---|
| `metric_improvement_delta` | 0.50 | Change in average quality metric (completeness, consistency, validity, uniqueness) |
| `issue_resolution_delta` | 0.30 | Fraction of issues resolved this step |
| `efficiency_bonus` | 0.02 × steps_remaining_ratio | Small bonus for resolving issues quickly |
| `penalty` | variable | −0.02 for failed actions, −0.05 for timeout |

**Key properties:**
- **Dense signal:** Every step that improves the dataset yields a positive reward
- **No reward hacking:** `inspect_column` actions yield exactly 0.0 reward (no state change)
- **Timeout penalty:** Exceeding `max_steps` yields a −0.05 penalty on the terminal reward
- **Efficiency incentive:** Solving issues in fewer steps earns a small bonus

---

## Setup & Usage

### Local installation
```bash
git clone https://huggingface.co/spaces/your-username/dataquality-openenv
cd dataquality-openenv
pip install -r requirements.txt
```

### Python API
```python
from dataquality_env import DataQualityEnv, Action, ActionType

env = DataQualityEnv(task_id="task_hard", seed=42)
obs = env.reset()

print(obs.task_description)
print(f"Issues: {[(i.issue_id, i.description) for i in obs.issue_registry]}")

# Fix missing patient IDs
action = Action(
    action_type=ActionType.DROP_ROWS,
    parameters={"column": "patient_id", "condition": "is_null"}
)
obs, reward, done, info = env.step(action)
print(f"Reward: {reward.total:+.4f}")
print(f"Result: {obs.last_action_result}")

# Get full internal state
state = env.state()
print(f"Score so far: {state.score}")
```

### HTTP API (after docker run or HF Space)
```bash
# Create a session
curl -X POST http://localhost:7860/env/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "task_medium", "seed": 42}'
# → {"session_id": "abc-123", "observation": {...}}

# Step
curl -X POST http://localhost:7860/env/step \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "abc-123",
    "action": {
      "action_type": "fix_type",
      "parameters": {"column": "revenue", "target_type": "float", "strip_chars": "$,"}
    }
  }'
# → {"observation": {...}, "reward": {...}, "done": false, "info": {...}}

# Get state
curl http://localhost:7860/env/state/abc-123
```

### Docker
```bash
docker build -t dataquality-env .
docker run -p 7860:7860 dataquality-env

# Interactive docs at: http://localhost:7860/docs
```

### Run baseline agent
```bash
export OPENAI_API_KEY=your_key_here
python baseline/run_baseline.py

# Optional: use a different model
BASELINE_MODEL=gpt-4o python baseline/run_baseline.py
```

---

## Baseline Scores

Reproducible scores using `seed=42`, model `gpt-4o-mini`:

| Task | Difficulty | Baseline Score | Oracle Score |
|---|---|---|---|
| `task_easy` | Easy | ~0.78 | 1.0000 |
| `task_medium` | Medium | ~0.72 | 1.0000 |
| `task_hard` | Hard | ~0.55 | 1.0000 |
| **Average** | | **~0.68** | **1.0000** |

*Oracle scores obtained by running the deterministic optimal action sequence (verified in tests).*

The gap between baseline and oracle on the hard task reflects the genuine difficulty of coordinating 7 simultaneous issue types under a step budget.

---

## Project Structure

```
dataquality-env/
├── dataquality_env/
│   ├── __init__.py         # Package exports
│   ├── env.py              # DataQualityEnv — step/reset/state
│   ├── models.py           # Typed Pydantic models (Action, Observation, Reward, EpisodeState)
│   └── tasks.py            # Dataset generators + graders for all 3 tasks
├── baseline/
│   └── run_baseline.py     # OpenAI-client baseline agent
├── tests/
│   └── test_env.py         # 30 tests covering spec compliance, actions, graders, rewards
├── app.py                  # FastAPI server for HF Spaces / HTTP API
├── openenv.yaml            # OpenEnv metadata
├── Dockerfile              # Container definition
├── requirements.txt
├── setup.py
└── README.md
```

---

## OpenEnv Spec Compliance

- ✅ Typed `Action`, `Observation`, `Reward`, `EpisodeState` Pydantic models
- ✅ `step(action)` → `(Observation, Reward, bool, Dict)`
- ✅ `reset()` → `Observation`
- ✅ `state()` → `EpisodeState`
- ✅ `openenv.yaml` with full metadata
- ✅ 3 tasks with difficulty progression (easy → medium → hard)
- ✅ Graders produce scores in `[0.0, 1.0]`, deterministic, seed-reproducible
- ✅ Meaningful partial-progress reward function (dense, not sparse)
- ✅ Baseline inference script using OpenAI API client
- ✅ Dockerfile + HF Spaces deployment

---

## License

MIT
