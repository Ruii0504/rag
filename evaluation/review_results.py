"""Print compact actual outputs and their evidence for the primary assistant's review."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--run-dir", default="evaluation/runs/2026-09-25-baseline")
    args = parser.parse_args()
    cases = {row["case_id"]: row for row in [json.loads(line) for line in (ROOT / "evaluation/regression-v1/regression-100.jsonl").read_text(encoding="utf-8").splitlines()]}
    for path in sorted((ROOT / args.run_dir / "traces").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        number = int(row["case_id"].split("-")[-1])
        if not args.start <= number <= args.end:
            continue
        case = cases[row["case_id"]]
        print(f"\n=== {row['case_id']} {case['expected_behavior']} ===")
        print("QUESTION:", case["question"])
        if case["history"]:
            print("HISTORY:", json.dumps(case["history"], ensure_ascii=False))
        print("GOLD:", case["groundtruthanswer"])
        print("ACTUAL:", row["final_answer"])
        print("COVERAGE:", json.dumps(row["strict_annotated_retrieval"], ensure_ascii=False))
        for call in row["calls"]:
            if call["stage"] in {"decompose", "judge", "validate"}:
                print(call["stage"].upper()+":", json.dumps(call.get("output", {"error": call.get("error_type")}), ensure_ascii=False))
            if args.detail and call["stage"] == "retrieval":
                for item in call.get("output", {}).get("evidences", []):
                    print(f"EVIDENCE local={item['reference_id']} {item['document_name'][:2]}:{item['chunk_index']} score={item['similarity']:.4f}")
                    print(item["content"])


if __name__ == "__main__":
    main()
