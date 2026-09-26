"""Validate annotation integrity against the snapshot and live read-only databases."""
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate(cases, chunks):
    require(len(cases) == 100, "Expected exactly 100 cases")
    require(Counter(row["tier"] for row in cases) == {"core": 60, "edge": 30, "high_risk": 10}, "Invalid tier quota")
    require(len({row["question"] for row in cases}) == 100, "Duplicate questions")
    require([row["case_id"] for row in cases] == [f"RAG-{i:03d}" for i in range(1, 101)], "Invalid case IDs")
    lookup = {row["chunk_id"]: row for row in chunks}
    require(len(chunks) == len(lookup) == 55, "Expected 55 unique persisted chunks")
    for chunk in chunks:
        require(hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest() == chunk["chunk_sha256"], "Chunk hash mismatch")
    for row in cases:
        label = row["case_id"]
        require(bool(row["question"].strip()) and bool(row["groundtruthanswer"].strip()), f"{label}: empty text")
        require(row["expected_behavior"] in {"answer", "partial_answer", "refuse", "clarify"}, f"{label}: invalid behavior")
        require(row["expected_intents"] and len(row["expected_intents"]) == len(set(row["expected_intents"])), f"{label}: missing/duplicate intents")
        expected_points = [] if row["expected_behavior"] == "clarify" else [group["claim"] for group in row["evidence_groups"]]
        require(row["required_points"] == expected_points, f"{label}: inconsistent required points")
        require(bool(row["required_behavior_checks"]), f"{label}: missing behavior checks")
        expected_chunks = set()
        for group in row["evidence_groups"]:
            require(bool(group["claim"]) and bool(group["any_of"]), f"{label}: empty evidence group")
            for option in group["any_of"]:
                refs = option["chunk_ids"]
                require(bool(refs) and all(ref in lookup for ref in refs), f"{label}: unknown chunk")
                require(set(option["document_ids"]) == {lookup[ref]["document_id"] for ref in refs}, f"{label}: document mismatch")
                require({quote["chunk_id"] for quote in option["quotes"]} == set(refs), f"{label}: missing quote")
                for quote in option["quotes"]:
                    require(len(quote["text"]) >= 10 and quote["text"] in lookup[quote["chunk_id"]]["content"], f"{label}: quote not found")
                expected_chunks.update(refs)
        require(set(row["groundtruthchunkid"]) == expected_chunks, f"{label}: flattened chunk IDs mismatch")
        require(len(row["groundtruthchunkid"]) == len(expected_chunks), f"{label}: duplicate chunk IDs")
        doc_ids = {lookup[ref]["document_id"] for ref in expected_chunks}
        require(set(row["groundtruthcumentid"]) == doc_ids, f"{label}: flattened document IDs mismatch")
        require(row["groundtruthdocumentid"] == row["groundtruthcumentid"], f"{label}: document alias mismatch")
        require(all(lookup[ref]["kb_id"] in row["kb_ids"] for ref in expected_chunks), f"{label}: KB scope mismatch")
        for turn in row["history"]:
            require(turn["role"] in {"user", "assistant"} and bool(turn["content"]), f"{label}: invalid history")
        if row["expected_behavior"] == "refuse":
            require(not expected_chunks and bool(row["missing_information"]), f"{label}: refusal must not invent positive evidence")
        else:
            require(bool(expected_chunks), f"{label}: missing supporting evidence")
        if row["expected_behavior"] in {"partial_answer", "clarify"}:
            require(bool(row["missing_information"]), f"{label}: missing unsupported or clarification item")
        if row["tier"] == "high_risk":
            require(bool(row["forbidden_claims"]), f"{label}: missing high-risk failure criteria")
    return {"case_count": len(cases), "tier_counts": dict(Counter(row["tier"] for row in cases)),
            "behavior_counts": dict(Counter(row["expected_behavior"] for row in cases)),
            "cited_chunk_count": len({ref for row in cases for ref in row["groundtruthchunkid"]}),
            "history_case_count": sum(bool(row["history"]) for row in cases)}


def validate_live(chunks, manifest):
    for doc in manifest["documents"]:
        text = (ROOT / doc["source_path"]).read_text(encoding="utf-8")
        require(hashlib.sha256(text.encode("utf-8")).hexdigest() == doc["content_hash"], "Source document has changed")
    app_path = ROOT / "rag-customer-service/data/app.db"
    with sqlite3.connect(app_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        live_documents = set(db.execute("SELECT id,kb_id,filename,content_hash,chunk_count FROM documents"))
    expected_documents = {(doc["id"], doc["kb_id"], doc["filename"], doc["content_hash"], doc["chunk_count"]) for doc in manifest["documents"]}
    require(live_documents == expected_documents, "Live document IDs, corpus scope or hashes changed; re-annotation required")
    chroma_path = ROOT / "rag-customer-service/data/chroma/chroma.sqlite3"
    with sqlite3.connect(chroma_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("""SELECT e.embedding_id,m.key,m.string_value,m.int_value
            FROM embeddings e JOIN embedding_metadata m ON m.id=e.id
            JOIN segments s ON s.id=e.segment_id JOIN collections c ON c.id=s.collection
            WHERE c.name='rag_chunks'""").fetchall()
    current = {}
    for chunk_id, key, string_value, int_value in rows:
        current.setdefault(chunk_id, {})[key] = string_value if string_value is not None else int_value
    require(set(current) == {chunk["chunk_id"] for chunk in chunks}, "Live corpus IDs changed")
    for chunk in chunks:
        meta = current[chunk["chunk_id"]]
        require(meta["chroma:document"] == chunk["content"], "Live chunk text changed")
        require(meta["document_id"] == chunk["document_id"] and meta["kb_id"] == chunk["kb_id"], "Live chunk metadata changed")


def main():
    cases = read_jsonl(HERE / "regression-100.jsonl")
    chunks = read_jsonl(HERE / "chunks.snapshot.jsonl")
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    require(cases == json.loads((HERE / "regression-100.json").read_text(encoding="utf-8")), "JSON and JSONL differ")
    with (HERE / "regression-100.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == len(cases), "CSV row count mismatch")
    for original, exported in zip(cases, csv_rows, strict=True):
        for key, value in exported.items():
            actual = json.loads(value) if isinstance(original[key], (list, dict)) else value
            require(actual == original[key], f"CSV mismatch: {original['case_id']} / {key}")
    for filename, key in [("regression-100.jsonl", "dataset_sha256"), ("chunks.snapshot.jsonl", "snapshot_sha256")]:
        require(hashlib.sha256((HERE / filename).read_bytes()).hexdigest() == manifest[key], f"{filename}: file hash mismatch")
    report = validate(cases, chunks)
    validate_live(chunks, manifest)
    report.update({"validation_status": "passed", "live_corpus_matches_snapshot": True,
                   "model_evaluation_run": False, "semantic_gold_review": "agent_reviewed_pending_user_signoff"})
    (HERE / "validation-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
