"""Verify report provenance, frozen inputs, and all 100 first-attempt records."""
import argparse
import json
import dataclasses
from datetime import datetime, timezone

from run_regression import DATA, ROOT, coverage, dump, plain, sha
from rag_customer_service.config import Settings

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", default="evaluation/runs/2026-09-25-baseline")
args = parser.parse_args()
run = (ROOT / args.run_dir).resolve()
if not run.is_relative_to(ROOT / "evaluation"):
    raise ValueError("Run directory must be inside evaluation")
manifest = json.loads((run / "run-manifest.json").read_text(encoding="utf-8"))
identity = manifest["identity"]
assert sha(DATA / "regression-100.jsonl") == identity["dataset_sha256"]
assert sha(DATA / "chunks.snapshot.jsonl") == identity["snapshot_sha256"]
assert sha(run / "rubric.md") == identity["rubric_sha256"]
assert {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / "rag-customer-service/src").rglob("*.py"))} == identity["source_sha256"]
settings = Settings.from_env(ROOT / "rag-customer-service")
assert {field.name: plain(getattr(settings, field.name)) for field in dataclasses.fields(settings)
        if field.name != "qwen_api_key"} == identity["settings"]
cases = [json.loads(line) for line in (DATA / "regression-100.jsonl").read_text(encoding="utf-8").splitlines()]
rows = json.loads((run / "scores.json").read_text(encoding="utf-8"))
assert len(cases) == len(rows) == 100
assert {row["case_id"] for row in rows} == {case["case_id"] for case in cases}
assert len(list((run / "traces").glob("*.json"))) == 100
rows_by_id = {row["case_id"]: row for row in rows}
for case in cases:
    path = run / "traces" / (case["case_id"] + ".json")
    trace = json.loads(path.read_text(encoding="utf-8"))
    row = rows_by_id[case["case_id"]]
    assert trace["question"] == case["question"] and trace["history"] == case["history"]
    assert row["trace_sha256"] == sha(path)
    assert row["final_answer"] == trace["final_answer"]
    assert row["strict_retrieval"] == coverage(case, trace["retrieved_chunk_ids"])
    if trace["execution_status"] == "error":
        assert all(score is None for score in row["scores"].values())
        assert not row["overall_pass"]
    else:
        assert trace["final_answer"].strip()
        assert all(isinstance(score, int) and 1 <= score <= 5 for score in row["scores"].values())
    for call in trace["calls"]:
        if call["stage"] == "retrieval" and "output" in call:
            assert len(call["output"]["evidences"]) <= 7
metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
assert metrics["overall_pass_count"] == sum(row["overall_pass"] for row in rows)
assert metrics["strict_retrieval"]["denominator"] == 93
for name, average in metrics["score_means"].items():
    values = [row["scores"][name] for row in rows if row["scores"][name] is not None]
    assert abs(average - sum(values) / len(values)) < 1e-12
provenance_path = run / "provenance.json"
provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else None
if provenance:
    inherited = set(provenance["inherited_success_case_ids"])
    executed = set(provenance["executed_with_fixed_code_case_ids"])
    assert not inherited & executed
    assert inherited | executed == {case["case_id"] for case in cases}
    source_run = ROOT / provenance["source_run"]
    assert sha(source_run / "run-manifest.json") == provenance["source_manifest_sha256"]
    for case_id in inherited:
        assert sha(run / "traces" / f"{case_id}.json") == sha(source_run / "traces" / f"{case_id}.json")
report = {"verified_at": datetime.now(timezone.utc).isoformat(), "cases": 100,
          "application_source_files_unchanged": len(identity["source_sha256"]),
          "dataset_snapshot_rubric_unchanged": True, "trace_score_metrics_consistent": True,
          "final_execution_errors": [row["case_id"] for row in rows if row["execution_status"] == "error"],
          "composite_result_set": bool(provenance),
          "inherited_success_cases": len(provenance["inherited_success_case_ids"]) if provenance else 0,
          "fixed_code_execution_cases": len(provenance["executed_with_fixed_code_case_ids"]) if provenance else 100}
dump(run / "verification.json", report)
print(json.dumps(report, ensure_ascii=False, indent=2))
