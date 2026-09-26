"""Inspect observed candidate positions without any additional model requests."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "evaluation/runs/2026-09-25-baseline"
cases = {r["case_id"]: r for r in [json.loads(line) for line in (ROOT / "evaluation/regression-v1/regression-100.jsonl").read_text(encoding="utf-8").splitlines()]}
report = []
for path in sorted((RUN / "traces").glob("*.json")):
    trace = json.loads(path.read_text(encoding="utf-8"))
    case = cases[trace["case_id"]]
    if not trace["strict_annotated_retrieval"] or trace["strict_annotated_retrieval"]["complete"]:
        continue
    searches = [c for c in trace["calls"] if c["stage"] == "vector_search" and "output" in c]
    retrievals = [c for c in trace["calls"] if c["stage"] == "retrieval" and "output" in c]
    groups = []
    for i, group in enumerate(case["evidence_groups"]):
        if trace["strict_annotated_retrieval"]["covered_groups"][i]:
            continue
        choices = {ref for option in group["any_of"] for ref in option["chunk_ids"]}
        observations = []
        for search, retrieval in zip(searches, retrievals, strict=True):
            ranks = sorted(search["output"], key=lambda r: (r["lexical_relevance"], r["similarity"]), reverse=True)
            for raw_rank, item in enumerate(search["output"], 1):
                if item["chunk_id"] in choices:
                    observations.append({"query": retrieval["input"]["query"], "chunk_id": item["chunk_id"],
                        "vector_rank": raw_rank, "lexical_rank": next(j for j, r in enumerate(ranks, 1) if r["chunk_id"] == item["chunk_id"]),
                        "similarity": item["similarity"], "threshold_pass": item["similarity"] >= 0.5})
        groups.append({"group_index": i, "claim": group["claim"], "observations": observations})
    report.append({"case_id": case["case_id"], "missing_groups": groups})
(RUN / "retrieval-diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
