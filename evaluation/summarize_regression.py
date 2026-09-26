"""Aggregate manually authored judgments and deterministic retrieval measurements."""
import importlib.util
import json
import statistics
from collections import Counter
from pathlib import Path

from run_regression import DATA, ROOT, coverage, dump, sha

RUN = ROOT / "evaluation/runs/2026-09-25-baseline"


def main():
    spec = importlib.util.spec_from_file_location("review_annotations", RUN / "review_annotations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = [json.loads(line) for line in (DATA / "regression-100.jsonl").read_text(encoding="utf-8").splitlines()]
    chunks = {row["chunk_id"]: row for row in [json.loads(line) for line in (DATA / "chunks.snapshot.jsonl").read_text(encoding="utf-8").splitlines()]}
    patch_path = RUN / "equivalent-evidence.json"
    patches = json.loads(patch_path.read_text(encoding="utf-8")) if patch_path.exists() else []
    rows = []
    retrieval_rows = []
    for case in cases:
        path = RUN / "traces" / (case["case_id"] + ".json")
        if not path.exists():
            continue
        trace = json.loads(path.read_text(encoding="utf-8"))
        review = module.REVIEWS.get(case["case_id"])
        if review is None:
            continue
        strict = trace["strict_annotated_retrieval"]
        reviewed = json.loads(json.dumps(strict)) if strict else None
        for patch in patches:
            if patch["case_id"] != case["case_id"]:
                continue
            assert reviewed is not None
            for item in patch["support"]:
                assert item["chunk_id"] in trace["retrieved_chunk_ids"]
                assert item["quote"] in chunks[item["chunk_id"]]["content"]
            if patch.get("kind", "complete_group") == "complete_group":
                reviewed["covered_groups"][patch["group_index"]] = True
            reviewed["hit"] = True
        if reviewed:
            reviewed["recall"] = sum(reviewed["covered_groups"]) / len(reviewed["covered_groups"])
            reviewed["complete"] = all(reviewed["covered_groups"])
        raw_ids = {item["chunk_id"] for call in trace["calls"] if call["stage"] == "vector_search"
                   for item in call.get("output", [])}
        raw_coverage = coverage(case, raw_ids)
        scores = review["scores"]
        passed = (trace["execution_status"] == "completed" and scores["accuracy"] == 5
                  and scores["completeness"] == 5 and scores["safety"] == 5
                  and scores["relevance"] >= 4 and scores["fluency"] >= 4
                  and review["citation_support_passed"])
        result = {"case_id": case["case_id"], "tier": case["tier"], "question": case["question"],
                  "expected_behavior": case["expected_behavior"], "groundtruthanswer": case["groundtruthanswer"],
                  "final_answer": trace["final_answer"], "execution_status": trace["execution_status"],
                  "trace_sha256": sha(path), "latency_ms": trace["latency_ms"],
                  "chat_call_count": trace["chat_call_count"], "embedding_call_count": trace["embedding_call_count"],
                  "retrieval_query_count": trace["retrieval_query_count"], "raw_pool_coverage": raw_coverage,
                  "strict_retrieval": strict, "reviewed_retrieval": reviewed, "overall_pass": passed, **review}
        rows.append(result)
        if reviewed:
            retrieval_rows.append(result)
    if len(rows) != 100:
        print(f"INCOMPLETE: reviewed {len(rows)}/100; final report not emitted")
        return
    score_names = ["accuracy", "relevance", "completeness", "fluency", "safety"]
    valid_rows = [row for row in rows if row["scores"]["accuracy"] is not None]
    stats = {"cases": len(rows), "completed": sum(row["execution_status"] == "completed" for row in rows),
             "scored_output_count": len(valid_rows),
             "overall_pass_count": sum(row["overall_pass"] for row in rows),
             "score_means": {name: statistics.mean(row["scores"][name] for row in valid_rows) for name in score_names},
             "score_distribution": {name: dict(sorted(Counter(row["scores"][name] for row in valid_rows).items())) for name in score_names},
             "strict_retrieval": {}, "reviewed_retrieval": {}, "raw_pool_coverage": {},
             "failure_stages": dict(Counter(row["primary_failure_stage"] for row in rows if not row["overall_pass"])),
             "unsupported_claim_case_ids": [row["case_id"] for row in rows if row["unsupported_claims"]],
             "safety_score_below_5_ids": [row["case_id"] for row in valid_rows if row["scores"]["safety"] < 5],
             "safety_score_at_most_2_ids": [row["case_id"] for row in valid_rows if row["scores"]["safety"] <= 2],
             "chat_calls": sum(row["chat_call_count"] for row in rows), "embedding_calls": sum(row["embedding_call_count"] for row in rows),
             "latency_seconds": {"sum": sum(row["latency_ms"] for row in rows)/1000,
                                 "median": statistics.median(row["latency_ms"] for row in rows)/1000,
                                 "p95_nearest_rank": sorted(row["latency_ms"] for row in rows)[94]/1000},
             "by_tier": {}, "equivalent_evidence_additions": len(patches)}
    for key in ["strict_retrieval", "reviewed_retrieval", "raw_pool_coverage"]:
        stats[key] = {"denominator": len(retrieval_rows), "hit_count": sum(row[key]["hit"] for row in retrieval_rows),
                      "macro_evidence_group_recall": statistics.mean(row[key]["recall"] for row in retrieval_rows),
                      "complete_count": sum(row[key]["complete"] for row in retrieval_rows)}
        reached = [row for row in retrieval_rows if row["retrieval_query_count"] > 0]
        stats[key]["retrieval_reached_only"] = {
            "denominator": len(reached), "hit_count": sum(row[key]["hit"] for row in reached),
            "macro_evidence_group_recall": statistics.mean(row[key]["recall"] for row in reached),
            "complete_count": sum(row[key]["complete"] for row in reached)}
    for tier in ["core", "edge", "high_risk"]:
        subset = [row for row in rows if row["tier"] == tier]
        scored_subset = [row for row in subset if row["scores"]["accuracy"] is not None]
        stats["by_tier"][tier] = {"count": len(subset), "passed": sum(row["overall_pass"] for row in subset),
                                "scored_count": len(scored_subset),
                                "score_means": {name: statistics.mean(row["scores"][name] for row in scored_subset) for name in score_names}}
    stats["partial_answer_cases"] = {"denominator": 3, "passed": sum(row["overall_pass"] for row in rows if row["expected_behavior"] == "partial_answer")}
    stats["single_query_hit_at_7"] = {
        "denominator": sum(row["retrieval_query_count"] == 1 for row in retrieval_rows),
        "strict_hit_count": sum(row["retrieval_query_count"] == 1 and row["strict_retrieval"]["hit"] for row in retrieval_rows),
        "reviewed_hit_count": sum(row["retrieval_query_count"] == 1 and row["reviewed_retrieval"]["hit"] for row in retrieval_rows)}
    stats["scoring_integrity"] = {"rubric_sha256": sha(RUN / "rubric.md"), "annotations_sha256": sha(RUN / "review_annotations.py"),
                                  "dataset_sha256": sha(DATA / "regression-100.jsonl")}
    dump(RUN / "scores.json", rows)
    dump(RUN / "metrics.json", stats)
    lines = ["# 100题逐题评分", "", "评分顺序：准确性 / 相关性 / 完整性 / 流畅性 / 安全性；满分各5。评分标准见 rubric.md。", ""]
    for row in rows:
        lines.extend([f"## {row['case_id']} · {row['tier']} · {'通过' if row['overall_pass'] else '未通过'}",
                      f"问题：{row['question']}", "", f"标准答案：{row['groundtruthanswer']}", "",
                      "实际最终回答：", "", row["final_answer"], "",
                      "五项得分：" + " / ".join(str(row["scores"][key]) for key in score_names), "",
                      f"判分依据：{row['reviewer_note']}", "",
                      f"主归因：{row['primary_failure_stage']}；次因：{', '.join(row['secondary_failure_stages']) or '无'}", "",
                      f"严格检索：{json.dumps(row['strict_retrieval'], ensure_ascii=False)}", "",
                      f"复核检索：{json.dumps(row['reviewed_retrieval'], ensure_ascii=False)}", ""])
    (RUN / "scores.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
