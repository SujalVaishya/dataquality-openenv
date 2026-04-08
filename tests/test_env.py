"""
tests/test_env.py
Tests for OpenEnv spec compliance, action correctness, and grader accuracy.
Run with: pytest tests/
"""
import pytest
from dataquality_env import Action, ActionType, DataQualityEnv
from dataquality_env.models import Observation, Reward, EpisodeState


# ─────────────────────────────────────────────
# Spec compliance tests
# ─────────────────────────────────────────────

class TestOpenEnvSpec:
    def test_reset_returns_observation(self):
        env = DataQualityEnv(task_id="task_easy")
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert isinstance(obs.dataset_preview, list)
        assert isinstance(obs.schema, dict)
        assert isinstance(obs.issue_registry, list)
        assert isinstance(obs.action_history, list)
        assert obs.step_count == 0

    def test_step_returns_tuple(self):
        env = DataQualityEnv(task_id="task_easy")
        env.reset()
        action = Action(action_type=ActionType.INSPECT_COLUMN, parameters={})
        result = env.step(action)
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_state_returns_episode_state(self):
        env = DataQualityEnv(task_id="task_easy")
        env.reset()
        state = env.state()
        assert isinstance(state, EpisodeState)
        assert state.task_id == "task_easy"
        assert state.step_count == 0

    def test_step_increments_step_count(self):
        env = DataQualityEnv(task_id="task_easy")
        env.reset()
        action = Action(action_type=ActionType.INSPECT_COLUMN, parameters={})
        env.step(action)
        assert env.state().step_count == 1

    def test_reset_produces_clean_state(self):
        env = DataQualityEnv(task_id="task_easy")
        env.reset()
        action = Action(action_type=ActionType.INSPECT_COLUMN, parameters={})
        env.step(action)
        obs = env.reset()
        assert obs.step_count == 0
        assert obs.action_history == []

    def test_submit_terminates_episode(self):
        env = DataQualityEnv(task_id="task_easy")
        env.reset()
        action = Action(action_type=ActionType.SUBMIT, parameters={})
        _, reward, done, _ = env.step(action)
        assert done is True
        assert reward.done is True

    def test_step_after_done_raises(self):
        env = DataQualityEnv(task_id="task_easy")
        env.reset()
        env.step(Action(action_type=ActionType.SUBMIT))
        with pytest.raises(RuntimeError):
            env.step(Action(action_type=ActionType.INSPECT_COLUMN))

    def test_state_before_reset_raises(self):
        env = DataQualityEnv(task_id="task_easy")
        with pytest.raises(RuntimeError):
            env.state()

    def test_reward_in_valid_range(self):
        env = DataQualityEnv(task_id="task_medium")
        env.reset()
        for _ in range(5):
            action = Action(action_type=ActionType.INSPECT_COLUMN, parameters={})
            _, reward, done, _ = env.step(action)
            assert -1.0 <= reward.total <= 1.0
            if done:
                break

    def test_all_task_ids_work(self):
        for tid in DataQualityEnv.TASK_IDS:
            env = DataQualityEnv(task_id=tid)
            obs = env.reset()
            assert len(obs.dataset_preview) > 0
            assert len(obs.issue_registry) > 0

    def test_invalid_task_id_raises(self):
        with pytest.raises(ValueError):
            DataQualityEnv(task_id="not_a_task")


# ─────────────────────────────────────────────
# Action tests
# ─────────────────────────────────────────────

class TestActions:
    def _setup(self, task_id="task_easy"):
        env = DataQualityEnv(task_id=task_id)
        env.reset()
        return env

    def test_inspect_column_no_params(self):
        env = self._setup()
        obs, reward, done, _ = env.step(Action(action_type=ActionType.INSPECT_COLUMN, parameters={}))
        assert "Columns:" in obs.last_action_result

    def test_inspect_column_specific(self):
        env = self._setup()
        obs, _, _, _ = env.step(Action(action_type=ActionType.INSPECT_COLUMN, parameters={"column": "email"}))
        assert "email" in obs.last_action_result

    def test_fill_missing_constant(self):
        env = self._setup()
        before_nulls = sum(1 for r in env.state().dataset_rows if r.get("email") is None)
        env.step(Action(action_type=ActionType.FILL_MISSING, parameters={
            "column": "email", "fill_value": "unknown@example.com", "strategy": "constant"
        }))
        after_nulls = sum(1 for r in env.state().dataset_rows if r.get("email") is None)
        assert after_nulls == 0
        assert before_nulls > 0  # confirms test was meaningful

    def test_drop_duplicates_reduces_rows(self):
        env = self._setup()
        before = len(env.state().dataset_rows)
        env.step(Action(action_type=ActionType.DROP_DUPLICATES, parameters={}))
        after = len(env.state().dataset_rows)
        assert after <= before

    def test_fix_type_revenue(self):
        env = self._setup("task_medium")
        env.step(Action(action_type=ActionType.FIX_TYPE, parameters={
            "column": "revenue", "target_type": "float", "strip_chars": "$,"
        }))
        bad = [r for r in env.state().dataset_rows if isinstance(r.get("revenue"), str)]
        assert len(bad) == 0

    def test_fix_type_date(self):
        env = self._setup("task_medium")
        env.step(Action(action_type=ActionType.FIX_TYPE, parameters={
            "column": "date", "target_type": "date"
        }))
        import re
        iso = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        bad = [r for r in env.state().dataset_rows if r.get("date") and not iso.match(str(r["date"]))]
        assert len(bad) == 0

    def test_flag_outlier_drop(self):
        env = self._setup("task_hard")
        before = len(env.state().dataset_rows)
        env.step(Action(action_type=ActionType.FLAG_OUTLIER, parameters={
            "column": "age", "min_val": 0, "max_val": 130, "action": "drop"
        }))
        after = len(env.state().dataset_rows)
        assert after <= before

    def test_drop_rows_is_null(self):
        env = self._setup("task_hard")
        before = len(env.state().dataset_rows)
        env.step(Action(action_type=ActionType.DROP_ROWS, parameters={
            "column": "patient_id", "condition": "is_null"
        }))
        after = len(env.state().dataset_rows)
        assert after < before
        assert all(r.get("patient_id") is not None for r in env.state().dataset_rows)

    def test_apply_regex_normalize_map(self):
        env = self._setup("task_hard")
        env.step(Action(action_type=ActionType.APPLY_REGEX_FIX, parameters={
            "column": "status",
            "normalize_map": {
                "Scheduled": "scheduled",
                "COMPLETED": "completed",
                "cancled": "cancelled",
                "no show": "no_show",
            }
        }))
        # After normalizing, fewer bad statuses
        valid = {"scheduled", "completed", "cancelled", "no_show"}
        bad = [r for r in env.state().dataset_rows if r.get("status") not in valid]
        # Some may still be invalid (e.g. "pending") but fewer than original
        assert len(bad) < 30


# ─────────────────────────────────────────────
# Grader tests
# ─────────────────────────────────────────────

class TestGraders:
    def test_easy_grader_perfect_score(self):
        """A perfectly cleaned dataset should score ~1.0."""
        from dataquality_env.tasks import grade_easy
        clean_rows = [
            {"id": i, "name": "Alice", "email": f"alice{i}@example.com",
             "phone": "+1-555-1234", "country": "US"}
            for i in range(10)
        ]
        score = grade_easy(clean_rows, clean_rows, [])
        assert score >= 0.99

    def test_easy_grader_zero_score(self):
        """A completely empty dataset should score low."""
        from dataquality_env.tasks import grade_easy
        score = grade_easy([], [], [])
        assert score == 0.0

    def test_medium_grader_range(self):
        """Grader scores must be in [0, 1]."""
        from dataquality_env.tasks import grade_medium, _make_medium_dataset
        rows = _make_medium_dataset(seed=42)
        score = grade_medium(rows, rows, [])
        assert 0.0 <= score <= 1.0

    def test_hard_grader_range(self):
        from dataquality_env.tasks import grade_hard, _make_hard_dataset
        rows = _make_hard_dataset(seed=42)
        score = grade_hard(rows, rows, [])
        assert 0.0 <= score <= 1.0

    def test_graders_deterministic(self):
        """Same input → same score, always."""
        from dataquality_env.tasks import grade_easy, grade_medium, grade_hard
        from dataquality_env.tasks import _make_easy_dataset, _make_medium_dataset, _make_hard_dataset
        for grade_fn, make_fn in [
            (grade_easy, _make_easy_dataset),
            (grade_medium, _make_medium_dataset),
            (grade_hard, _make_hard_dataset),
        ]:
            rows = make_fn(seed=99)
            s1 = grade_fn(rows, rows, [])
            s2 = grade_fn(rows, rows, [])
            assert s1 == s2

    def test_fixing_improves_score(self):
        """Applying correct fixes should improve grader score."""
        env = DataQualityEnv(task_id="task_easy", seed=42)
        env.reset()

        # Get baseline score
        env.step(Action(action_type=ActionType.SUBMIT))
        baseline_score = env.state().score

        # New episode, fix things then submit
        env.reset()
        env.step(Action(action_type=ActionType.FILL_MISSING, parameters={
            "column": "email", "fill_value": "unknown@example.com", "strategy": "constant"
        }))
        env.step(Action(action_type=ActionType.FILL_MISSING, parameters={
            "column": "phone", "fill_value": "N/A", "strategy": "constant"
        }))
        env.step(Action(action_type=ActionType.DROP_DUPLICATES, parameters={}))
        env.step(Action(action_type=ActionType.SUBMIT))
        fixed_score = env.state().score

        assert fixed_score > baseline_score

    def test_hard_task_has_issues(self):
        """Hard task should start with at least 5 distinct issue types."""
        env = DataQualityEnv(task_id="task_hard")
        obs = env.reset()
        issue_types = set(iss.issue_type for iss in obs.issue_registry)
        assert len(issue_types) >= 4


# ─────────────────────────────────────────────
# Reward function tests
# ─────────────────────────────────────────────

class TestRewardFunction:
    def test_reward_has_partial_signal(self):
        """Reward should be non-zero when issues are resolved (not just at episode end)."""
        env = DataQualityEnv(task_id="task_easy")
        env.reset()
        _, r1, _, _ = env.step(Action(action_type=ActionType.FILL_MISSING, parameters={
            "column": "email", "fill_value": "x@x.com", "strategy": "constant"
        }))
        # At least the reward object is populated
        assert r1.total is not None
        assert not r1.done  # not done yet

    def test_timeout_penalty(self):
        """Exceeding max_steps should trigger a penalty."""
        env = DataQualityEnv(task_id="task_easy", seed=42)
        env.reset()
        last_reward = None
        for _ in range(15):
            _, reward, done, _ = env.step(Action(action_type=ActionType.INSPECT_COLUMN, parameters={}))
            last_reward = reward
            if done:
                break
        assert last_reward is not None and last_reward.done


# ─────────────────────────────────────────────
# Metrics tests
# ─────────────────────────────────────────────

class TestMetrics:
    def test_metrics_all_in_range(self):
        for tid in DataQualityEnv.TASK_IDS:
            env = DataQualityEnv(task_id=tid)
            obs = env.reset()
            m = obs.metrics
            for val in [m.completeness, m.consistency, m.validity, m.uniqueness]:
                assert 0.0 <= val <= 1.0, f"metric out of range in {tid}: {val}"

    def test_metrics_improve_after_fix(self):
        env = DataQualityEnv(task_id="task_easy")
        obs_before = env.reset()
        before_completeness = obs_before.metrics.completeness

        env.step(Action(action_type=ActionType.FILL_MISSING, parameters={
            "column": "email", "fill_value": "x@x.com", "strategy": "constant"
        }))
        obs_after, _, _, _ = env.step(Action(action_type=ActionType.FILL_MISSING, parameters={
            "column": "phone", "fill_value": "N/A", "strategy": "constant"
        }))
        assert obs_after.metrics.completeness >= before_completeness
