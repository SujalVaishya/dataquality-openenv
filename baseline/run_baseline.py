#!/usr/bin/env python3
"""
baseline/run_baseline.py

Baseline agent that uses the OpenAI API client to run a model against
all three DataQuality tasks. Reads OPENAI_API_KEY from environment.

Usage:
    export OPENAI_API_KEY=your_key_here
    python baseline/run_baseline.py

    # Or with a different model:
    BASELINE_MODEL=gpt-4o python baseline/run_baseline.py

Expected reproducible scores (seed=42):
    task_easy:   ~0.72–0.85
    task_medium: ~0.65–0.80
    task_hard:   ~0.50–0.70
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from dataquality_env import Action, ActionType, DataQualityEnv

BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "gpt-4o-mini")
SEED = 42
MAX_RETRIES = 3

SYSTEM_PROMPT = """You are a data quality engineer. You are given a messy tabular dataset
and must fix data quality issues using the provided action API.

You must respond ONLY with a valid JSON object in this exact format:
{
  "action_type": "<one of: inspect_column, fill_missing, drop_duplicates, fix_type, flag_outlier, apply_regex_fix, drop_rows, rename_column, submit>",
  "parameters": { ... }
}

Action parameter guides:
- inspect_column: {"column": "col_name"} or {} to list all columns
- fill_missing: {"column": "col", "fill_value": "val", "strategy": "constant|forward_fill|mean"}
- drop_duplicates: {"subset": ["col1","col2"] or null for all columns, "keep": "first"}
- fix_type: {"column": "col", "target_type": "float|int|str|date|datetime", "strip_chars": "$,€"}
- flag_outlier: {"column": "col", "min_val": 0, "max_val": 130, "action": "drop|clip|flag"}
- apply_regex_fix: {"column": "col", "normalize_map": {"bad": "good"}} OR {"pattern": "...", "replacement": "..."}
- drop_rows: {"column": "col", "condition": "is_null|not_in|less_than|greater_than", "value": ...}
- rename_column: {"old_name": "old", "new_name": "new"}
- submit: {} (use when you believe all issues are resolved)

Strategy:
1. First inspect columns to understand the data
2. Address each issue systematically
3. Submit when done or near step limit"""


def build_user_message(obs: Dict[str, Any], step: int) -> str:
    """Build a descriptive user message from the observation."""
    issues = obs.get("issue_registry", [])
    open_issues = [iss for iss in issues if not iss.get("resolved", False)]
    metrics = obs.get("metrics", {})

    msg_parts = [
        f"Step {step}",
        f"Task: {obs['task_description'][:300]}",
        f"",
        f"Dataset: {len(obs['dataset_preview'])} rows shown (preview)",
        f"Schema: {json.dumps(obs['schema'])}",
        f"",
        f"Open issues ({len(open_issues)} of {len(issues)}):",
    ]
    for iss in open_issues[:6]:
        msg_parts.append(f"  - [{iss['issue_id']}] {iss['issue_type']} in '{iss.get('column','?')}': {iss['description']}")

    msg_parts += [
        f"",
        f"Metrics: completeness={metrics.get('completeness',0):.2f}, "
        f"consistency={metrics.get('consistency',0):.2f}, "
        f"validity={metrics.get('validity',0):.2f}, "
        f"uniqueness={metrics.get('uniqueness',0):.2f}",
        f"",
        f"Last action result: {obs.get('last_action_result', '')}",
        f"",
        f"Recent actions: {obs.get('action_history', [])[-3:]}",
        f"",
        f"Respond with your next action as JSON.",
    ]
    return "\n".join(msg_parts)


def call_llm(client: OpenAI, conversation: List[Dict], model: str) -> str:
    """Call the LLM and return the response text."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=conversation,
                temperature=0.2,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    return ""


def parse_action(text: str) -> Action:
    """Parse LLM output into an Action."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(text)
        return Action(**data)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  [!] Failed to parse action: {e}. Raw: {text[:200]}")
        # Fallback: inspect
        return Action(action_type=ActionType.INSPECT_COLUMN, parameters={})


def run_task(client: OpenAI, task_id: str, model: str, seed: int = 42) -> float:
    """Run one episode and return the final score."""
    print(f"\n{'='*60}")
    print(f"Running task: {task_id} | model: {model} | seed: {seed}")
    print(f"{'='*60}")

    env = DataQualityEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    obs_dict = obs.model_dump()

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = False
    step = 0
    cumulative_reward = 0.0

    while not done:
        user_msg = build_user_message(obs_dict, step)
        conversation.append({"role": "user", "content": user_msg})

        llm_response = call_llm(client, conversation, model)
        conversation.append({"role": "assistant", "content": llm_response})

        action = parse_action(llm_response)
        print(f"  Step {step+1}: {action.action_type} | params={json.dumps(action.parameters)[:80]}")

        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()
        cumulative_reward += reward.total

        print(f"    → reward={reward.total:+.4f} | metrics_overall="
              f"{obs.metrics.completeness:.2f}/{obs.metrics.consistency:.2f}/"
              f"{obs.metrics.validity:.2f}/{obs.metrics.uniqueness:.2f} | "
              f"result: {obs.last_action_result[:60] if obs.last_action_result else ''}")

        step += 1

    final_state = env.state()
    print(f"\n  ✓ Episode done. Final score: {final_state.score:.4f} | "
          f"Cumulative reward: {cumulative_reward:.4f} | Steps: {step}")
    return final_state.score


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    model = BASELINE_MODEL

    print(f"DataQuality Environment — Baseline Inference")
    print(f"Model: {model} | Seed: {SEED}")

    results = {}
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        try:
            score = run_task(client, task_id, model, seed=SEED)
            results[task_id] = score
        except Exception as e:
            print(f"  ERROR on {task_id}: {e}")
            results[task_id] = 0.0

    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    for task_id, score in results.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id:15s}: {score:.4f}  {bar}")
    avg = sum(results.values()) / len(results)
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}")

    # Write results to file for reproducibility
    output_path = Path(__file__).parent / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump({"model": model, "seed": SEED, "scores": results, "average": avg}, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
