"""
dataquality_env/tasks.py
Task definitions: dataset generators + programmatic graders.
Each task returns (dataset_rows, issue_registry, task_description, grader_fn).
"""
from __future__ import annotations

import copy
import random
from typing import Any, Callable, Dict, List, Tuple

from dataquality_env.models import IssueRecord


# ─────────────────────────────────────────────
# TASK 1: Easy — Basic Completeness Fix
# ─────────────────────────────────────────────

def _make_easy_dataset(seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
             "Iris", "Jack", "Karen", "Leo", "Mona", "Ned", "Olive"]
    domains = ["example.com", "company.org", "mail.net"]
    rows = []
    for i in range(30):
        name = rng.choice(names)
        email = f"{name.lower()}{i}@{rng.choice(domains)}" if rng.random() > 0.3 else None
        phone = f"+1-555-{rng.randint(1000,9999)}" if rng.random() > 0.25 else None
        rows.append({
            "id": i + 1,
            "name": name,
            "email": email,
            "phone": phone,
            "country": rng.choice(["US", "UK", "CA", "AU"]),
        })
    # Add 5 exact duplicates
    for _ in range(5):
        rows.append(copy.deepcopy(rng.choice(rows[:20])))
    rng.shuffle(rows)
    return rows


def _make_easy_issues(rows: List[Dict]) -> List[IssueRecord]:
    issues = []
    missing_email_rows = [i for i, r in enumerate(rows) if r.get("email") is None]
    missing_phone_rows = [i for i, r in enumerate(rows) if r.get("phone") is None]

    if missing_email_rows:
        issues.append(IssueRecord(
            issue_id="ISS-001",
            issue_type="missing",
            column="email",
            row_indices=missing_email_rows,
            severity="major",
            description=f"{len(missing_email_rows)} rows missing email address",
        ))
    if missing_phone_rows:
        issues.append(IssueRecord(
            issue_id="ISS-002",
            issue_type="missing",
            column="phone",
            row_indices=missing_phone_rows,
            severity="minor",
            description=f"{len(missing_phone_rows)} rows missing phone number",
        ))
    # Duplicates
    seen_ids = set()
    dup_rows = []
    for i, r in enumerate(rows):
        key = (r["name"], r.get("email"), r.get("phone"), r["country"])
        if key in seen_ids:
            dup_rows.append(i)
        else:
            seen_ids.add(key)
    if dup_rows:
        issues.append(IssueRecord(
            issue_id="ISS-003",
            issue_type="duplicate",
            row_indices=dup_rows,
            severity="major",
            description=f"{len(dup_rows)} exact duplicate rows detected",
        ))
    return issues


def grade_easy(current_rows: List[Dict], original_rows: List[Dict], action_history: List[str]) -> float:
    """
    Grade based on:
    - 40%: no missing emails
    - 30%: no missing phones (or sentinel filled)
    - 30%: no exact duplicates
    """
    if not current_rows:
        return 0.0

    missing_email = sum(1 for r in current_rows if not r.get("email"))
    missing_phone = sum(1 for r in current_rows if not r.get("phone"))

    email_score = 1.0 if missing_email == 0 else max(0.0, 1.0 - missing_email / max(len(current_rows), 1))
    phone_score = 1.0 if missing_phone == 0 else max(0.0, 1.0 - missing_phone / max(len(current_rows), 1))

    # Check duplicates
    seen = set()
    dup_count = 0
    for r in current_rows:
        key = (r.get("name"), r.get("email"), r.get("phone"), r.get("country"))
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
    dup_score = 1.0 if dup_count == 0 else max(0.0, 1.0 - dup_count / max(len(current_rows), 1))

    return round(0.4 * email_score + 0.3 * phone_score + 0.3 * dup_score, 4)


# ─────────────────────────────────────────────
# TASK 2: Medium — Type Errors & Format Violations
# ─────────────────────────────────────────────

def _make_medium_dataset(seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    bad_date_formats = [
        "2024/01/15", "15-01-2024", "Jan 15 2024", "2024-01-15",  # 2024-01-15 is correct ISO
        "01/15/2024", "15.01.2024",
    ]
    for i in range(40):
        # Revenue: should be float, some are strings like "$1,234.56"
        revenue_raw = rng.uniform(100, 50000)
        if rng.random() < 0.3:
            revenue = f"${revenue_raw:,.2f}"  # bad string format
        else:
            revenue = round(revenue_raw, 2)

        # Date: mixed formats
        date = rng.choice(bad_date_formats) if rng.random() < 0.4 else "2024-01-15"

        # Quantity: some negatives (bad data)
        qty = rng.randint(1, 100)
        if rng.random() < 0.15:
            qty = -qty

        rows.append({
            "transaction_id": f"TXN-{1000+i}" if rng.random() > 0.1 else f"TXN-{1000 + rng.randint(0, i)}",
            "date": date,
            "revenue": revenue,
            "quantity": qty,
            "product": rng.choice(["Widget", "Gadget", "Doohickey", "Thingamajig"]),
        })
    return rows


def _make_medium_issues(rows: List[Dict]) -> List[IssueRecord]:
    issues = []

    bad_revenue_rows = [i for i, r in enumerate(rows) if isinstance(r.get("revenue"), str)]
    if bad_revenue_rows:
        issues.append(IssueRecord(
            issue_id="ISS-101",
            issue_type="type_error",
            column="revenue",
            row_indices=bad_revenue_rows,
            severity="critical",
            description=f"{len(bad_revenue_rows)} revenue values stored as strings with currency symbols",
        ))

    bad_date_rows = [i for i, r in enumerate(rows)
                     if r.get("date") and r["date"] != "2024-01-15" and "/" in str(r["date"])]
    more_bad = [i for i, r in enumerate(rows)
                if r.get("date") and "Jan" in str(r.get("date", ""))]
    bad_date_rows = list(set(bad_date_rows + more_bad))
    if bad_date_rows:
        issues.append(IssueRecord(
            issue_id="ISS-102",
            issue_type="format",
            column="date",
            row_indices=bad_date_rows,
            severity="major",
            description=f"{len(bad_date_rows)} dates in non-ISO format",
        ))

    neg_qty_rows = [i for i, r in enumerate(rows) if isinstance(r.get("quantity"), int) and r["quantity"] < 0]
    if neg_qty_rows:
        issues.append(IssueRecord(
            issue_id="ISS-103",
            issue_type="outlier",
            column="quantity",
            row_indices=neg_qty_rows,
            severity="major",
            description=f"{len(neg_qty_rows)} rows with negative quantities",
        ))

    txn_ids = [r.get("transaction_id") for r in rows]
    seen_txn: set = set()
    dup_txn_rows = []
    for i, tid in enumerate(txn_ids):
        if tid in seen_txn:
            dup_txn_rows.append(i)
        else:
            seen_txn.add(tid)
    if dup_txn_rows:
        issues.append(IssueRecord(
            issue_id="ISS-104",
            issue_type="duplicate",
            column="transaction_id",
            row_indices=dup_txn_rows,
            severity="critical",
            description=f"{len(dup_txn_rows)} duplicate transaction IDs",
        ))

    return issues


def grade_medium(current_rows: List[Dict], original_rows: List[Dict], action_history: List[str]) -> float:
    """
    Grade:
    - 35%: revenue all numeric
    - 25%: dates all ISO format
    - 20%: no negative quantities
    - 20%: no duplicate transaction IDs
    """
    if not current_rows:
        return 0.0

    bad_revenue = sum(1 for r in current_rows if isinstance(r.get("revenue"), str))
    revenue_score = 1.0 if bad_revenue == 0 else max(0.0, 1.0 - bad_revenue / len(current_rows))

    import re
    iso_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    bad_dates = sum(1 for r in current_rows
                    if r.get("date") and not iso_pattern.match(str(r["date"])))
    date_score = 1.0 if bad_dates == 0 else max(0.0, 1.0 - bad_dates / len(current_rows))

    neg_qty = sum(1 for r in current_rows if isinstance(r.get("quantity"), (int, float)) and r["quantity"] < 0)
    qty_score = 1.0 if neg_qty == 0 else max(0.0, 1.0 - neg_qty / len(current_rows))

    seen_ids: set = set()
    dup_ids = 0
    for r in current_rows:
        tid = r.get("transaction_id")
        if tid in seen_ids:
            dup_ids += 1
        else:
            seen_ids.add(tid)
    dup_score = 1.0 if dup_ids == 0 else max(0.0, 1.0 - dup_ids / len(current_rows))

    return round(0.35 * revenue_score + 0.25 * date_score + 0.20 * qty_score + 0.20 * dup_score, 4)


# ─────────────────────────────────────────────
# TASK 3: Hard — Multi-Issue Production Dataset
# ─────────────────────────────────────────────

VALID_STATUSES = {"scheduled", "completed", "cancelled", "no_show"}
VALID_DEPARTMENTS = {"cardiology", "orthopedics", "neurology", "pediatrics", "oncology"}


def _make_hard_dataset(seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    bad_statuses = ["Scheduled", "COMPLETED", "cancled", "no show", "pending"]

    for i in range(60):
        patient_id = f"P{rng.randint(1000, 1099)}" if rng.random() > 0.15 else None
        appt_id = f"A{2000+i}" if rng.random() > 0.08 else f"A{2000 + rng.randint(0, i)}"

        # Multiple date format columns
        appt_date = "2024-03-15" if rng.random() > 0.4 else rng.choice(
            ["15/03/2024", "Mar 15 2024", "2024/03/15", "03-15-2024"]
        )
        created_ts = "2024-03-01T09:00:00" if rng.random() > 0.35 else rng.choice(
            ["2024-03-01 09:00", "03/01/2024 9:00 AM", "March 1 2024"]
        )

        age = rng.randint(0, 100)
        if rng.random() < 0.1:
            age = rng.choice([-5, 150, 999, -1])

        status = rng.choice(list(VALID_STATUSES)) if rng.random() > 0.25 else rng.choice(bad_statuses)
        department = rng.choice(list(VALID_DEPARTMENTS)) if rng.random() > 0.2 else rng.choice(
            ["Cardiology", "ORTHO", "neuro", "kids", "unknown_dept"]
        )

        rows.append({
            "appointment_id": appt_id,
            "patient_id": patient_id,
            "age": age,
            "appointment_date": appt_date,
            "created_timestamp": created_ts,
            "status": status,
            "department": department,
            "duration_minutes": rng.randint(15, 120),
        })

    # Add referential integrity violations: appointment with no patient
    for _ in range(5):
        rows.append({
            "appointment_id": f"A{3000 + rng.randint(0, 99)}",
            "patient_id": f"P{rng.randint(9000, 9099)}",  # patient IDs that don't exist in registry
            "age": rng.randint(20, 80),
            "appointment_date": "2024-03-15",
            "created_timestamp": "2024-03-01T09:00:00",
            "status": "scheduled",
            "department": "cardiology",
            "duration_minutes": 30,
        })
    rng.shuffle(rows)
    return rows


def _make_hard_issues(rows: List[Dict]) -> List[IssueRecord]:
    import re
    issues = []
    iso_date = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    iso_ts = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

    miss_pid = [i for i, r in enumerate(rows) if not r.get("patient_id")]
    if miss_pid:
        issues.append(IssueRecord(issue_id="ISS-201", issue_type="missing", column="patient_id",
            row_indices=miss_pid, severity="critical", description=f"{len(miss_pid)} appointments missing patient_id"))

    bad_date = [i for i, r in enumerate(rows) if r.get("appointment_date") and not iso_date.match(str(r["appointment_date"]))]
    if bad_date:
        issues.append(IssueRecord(issue_id="ISS-202", issue_type="format", column="appointment_date",
            row_indices=bad_date, severity="major", description=f"{len(bad_date)} appointment_dates not in ISO format"))

    bad_ts = [i for i, r in enumerate(rows) if r.get("created_timestamp") and not iso_ts.match(str(r["created_timestamp"]))]
    if bad_ts:
        issues.append(IssueRecord(issue_id="ISS-203", issue_type="format", column="created_timestamp",
            row_indices=bad_ts, severity="major", description=f"{len(bad_ts)} created_timestamps not in ISO format"))

    bad_age = [i for i, r in enumerate(rows) if isinstance(r.get("age"), int) and (r["age"] < 0 or r["age"] > 130)]
    if bad_age:
        issues.append(IssueRecord(issue_id="ISS-204", issue_type="outlier", column="age",
            row_indices=bad_age, severity="critical", description=f"{len(bad_age)} rows with invalid age values"))

    bad_status = [i for i, r in enumerate(rows) if r.get("status") not in VALID_STATUSES]
    if bad_status:
        issues.append(IssueRecord(issue_id="ISS-205", issue_type="format", column="status",
            row_indices=bad_status, severity="major", description=f"{len(bad_status)} rows with invalid status values"))

    bad_dept = [i for i, r in enumerate(rows) if r.get("department") not in VALID_DEPARTMENTS]
    if bad_dept:
        issues.append(IssueRecord(issue_id="ISS-206", issue_type="format", column="department",
            row_indices=bad_dept, severity="major", description=f"{len(bad_dept)} rows with invalid department values"))

    seen_appt: set = set()
    dup_appt = []
    for i, r in enumerate(rows):
        aid = r.get("appointment_id")
        if aid in seen_appt:
            dup_appt.append(i)
        else:
            seen_appt.add(aid)
    if dup_appt:
        issues.append(IssueRecord(issue_id="ISS-207", issue_type="duplicate", column="appointment_id",
            row_indices=dup_appt, severity="critical", description=f"{len(dup_appt)} duplicate appointment IDs"))

    return issues


def grade_hard(current_rows: List[Dict], original_rows: List[Dict], action_history: List[str]) -> float:
    """
    Grade across 7 dimensions (weighted):
    patient_id present (20%), appt_date ISO (15%), created_ts ISO (15%),
    valid age (15%), valid status (15%), valid dept (10%), unique appt_id (10%)
    """
    import re
    if not current_rows:
        return 0.0
    n = len(current_rows)

    iso_date = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    iso_ts = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

    miss_pid = sum(1 for r in current_rows if not r.get("patient_id"))
    pid_score = max(0.0, 1.0 - miss_pid / n)

    bad_date = sum(1 for r in current_rows if r.get("appointment_date") and not iso_date.match(str(r["appointment_date"])))
    date_score = max(0.0, 1.0 - bad_date / n)

    bad_ts = sum(1 for r in current_rows if r.get("created_timestamp") and not iso_ts.match(str(r["created_timestamp"])))
    ts_score = max(0.0, 1.0 - bad_ts / n)

    bad_age = sum(1 for r in current_rows if isinstance(r.get("age"), int) and (r["age"] < 0 or r["age"] > 130))
    age_score = max(0.0, 1.0 - bad_age / n)

    bad_status = sum(1 for r in current_rows if r.get("status") not in VALID_STATUSES)
    status_score = max(0.0, 1.0 - bad_status / n)

    bad_dept = sum(1 for r in current_rows if r.get("department") not in VALID_DEPARTMENTS)
    dept_score = max(0.0, 1.0 - bad_dept / n)

    seen: set = set()
    dup_ids = 0
    for r in current_rows:
        aid = r.get("appointment_id")
        if aid in seen:
            dup_ids += 1
        else:
            seen.add(aid)
    dup_score = max(0.0, 1.0 - dup_ids / n)

    return round(
        0.20 * pid_score + 0.15 * date_score + 0.15 * ts_score +
        0.15 * age_score + 0.15 * status_score + 0.10 * dept_score +
        0.10 * dup_score, 4
    )


# ─────────────────────────────────────────────
# Task registry
# ─────────────────────────────────────────────

TaskDef = Tuple[
    List[Dict],        # dataset rows
    List[IssueRecord], # initial issues
    str,               # task description
    Callable,          # grader function
    int,               # max_steps
]


def get_task(task_id: str, seed: int = 42) -> TaskDef:
    if task_id == "task_easy":
        rows = _make_easy_dataset(seed)
        issues = _make_easy_issues(rows)
        desc = (
            "TASK (Easy): A customer contacts CSV has missing email addresses, missing phone numbers, "
            "and exact duplicate rows. Your goal: fill all missing emails using a plausible placeholder "
            "(e.g. 'unknown@example.com'), fill missing phones with 'N/A', and remove all exact duplicates. "
            "Use inspect_column to understand each column, then fix issues one at a time. Submit when done."
        )
        return rows, issues, desc, grade_easy, 15

    elif task_id == "task_medium":
        rows = _make_medium_dataset(seed)
        issues = _make_medium_issues(rows)
        desc = (
            "TASK (Medium): A sales transactions dataset has: (1) revenue stored as strings with currency symbols "
            "like '$1,234.56' — convert to float, (2) dates in mixed non-ISO formats — standardize to YYYY-MM-DD, "
            "(3) negative quantity values — drop those rows or set to abs value, "
            "(4) duplicate transaction IDs — remove duplicate rows. Fix all issues systematically."
        )
        return rows, issues, desc, grade_medium, 20

    elif task_id == "task_hard":
        rows = _make_hard_dataset(seed)
        issues = _make_hard_issues(rows)
        desc = (
            "TASK (Hard): A healthcare appointments dataset has 7 types of issues: "
            "(1) missing patient_id — drop those rows, "
            "(2) appointment_date not in YYYY-MM-DD — standardize, "
            "(3) created_timestamp not in YYYY-MM-DDThh:mm:ss — standardize, "
            "(4) age outliers (negative or >130) — drop those rows, "
            f"(5) status not in {VALID_STATUSES} — fix or drop, "
            f"(6) department not in {VALID_DEPARTMENTS} — fix or drop, "
            "(7) duplicate appointment_id — remove duplicates. "
            "Preserve as many valid records as possible. Address all issue types."
        )
        return rows, issues, desc, grade_hard, 30

    else:
        raise ValueError(f"Unknown task_id: {task_id}. Choose from: task_easy, task_medium, task_hard")
