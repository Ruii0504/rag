"""Read-only compact evidence packet for manual evaluation; does not assign scores."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--start", type=int, required=True)
parser.add_argument("--end", type=int, required=True)
args = parser.parse_args()
cases = {row["case_id"]: row for row in map(json.loads, (ROOT / "evaluation/regression-v1/regression-100.jsonl").read_text(encoding="utf-8").splitlines())}
seen = set()
for number in range(args.start, args.end + 1):
    case_id = f"RAG-{number:03d}"
    path = ROOT / args.run_dir / "traces" / f"{case_id}.json"
    if not path.exists():
        continue
    trace = json.loads(path.read_text(encoding="utf-8"))
    case = cases[case_id]
    print(f"\n=== {case_id} {trace['execution_status']} ===")
    for key in ("question", "history", "groundtruthanswer", "required_points", "required_behavior_checks", "forbidden_claims"):
        print(key, json.dumps(case.get(key), ensure_ascii=False))
    print("ACTUAL", trace["final_answer"])
    print("COVERAGE", trace["strict_annotated_retrieval"])
    used = {int(number) for group in re.findall(r"\[([\d,，\s]+)\]", trace["final_answer"]) for number in re.findall(r"\d+", group)}
    for event in trace["events"]:
        if event["type"] == "error":
            print("ERROR", event["message"])
        if event["type"] == "evidence":
            for evidence in event["payload"].get("result", {}).get("evidences", []):
                if evidence["reference_id"] not in used:
                    continue
                chunk_id = f"{evidence['document_id']}:{evidence['chunk_index']}"
                print("CITED", evidence["reference_id"], chunk_id, evidence["heading_path"])
                if chunk_id not in seen:
                    print(evidence["content"])
                    seen.add(chunk_id)
    for call in trace["calls"]:
        if call["stage"] in {"judge", "validate"}:
            print(call["stage"], json.dumps(call.get("output", {}), ensure_ascii=False))
    print("MODEL_ERRORS", json.dumps([call for call in trace.get("model_calls", []) if call["status"] == "error"], ensure_ascii=False))
