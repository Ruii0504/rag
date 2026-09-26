"""Run the unchanged project agent with local observation wrappers and resumable output."""
import argparse
import dataclasses
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "evaluation/regression-v1"
sys.path.insert(0, str(ROOT / "rag-customer-service/src"))
sys.path.insert(0, str(DATA))


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def dump(path, value):
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coverage(case, ids):
    groups = case["evidence_groups"]
    if case["expected_behavior"] in {"refuse", "clarify"}:
        return None
    covered = [any(set(option["chunk_ids"]) <= set(ids) for option in group["any_of"]) for group in groups]
    return {"hit": bool(set(ids) & set(case["groundtruthchunkid"])), "covered_groups": covered,
            "recall": sum(covered) / len(covered), "complete": all(covered)}


class Observer:
    def __init__(self):
        self.record = None

    def wrap(self, owner, name, label, input_fn=None, output_fn=plain):
        original = getattr(owner, name)

        def observed(*args, **kwargs):
            started = time.perf_counter()
            entry = {"stage": label}
            if input_fn:
                entry["input"] = input_fn(args, kwargs)
            self.record["calls"].append(entry)
            try:
                result = original(*args, **kwargs)
                entry["output"] = output_fn(result)
                return result
            except Exception as error:
                entry["error_type"] = type(error).__name__
                raise
            finally:
                entry["latency_ms"] = round((time.perf_counter() - started) * 1000)

        setattr(owner, name, observed)

    def install(self, container):
        agent = container.agent
        retriever = agent._retriever
        qwen = retriever._qwen
        self.wrap(qwen, "chat", "chat", lambda args, kw: plain(args[0]))
        self.wrap(qwen, "embed_query", "embedding", lambda args, kw: args[0], lambda result: {"dimension": len(result)})
        self.wrap(retriever._vector_store, "search", "vector_search",
                  lambda args, kw: {"kb_ids": list(args[1]), "requested_k": kw.get("top_k")},
                  lambda rows: [{"chunk_id": item.chunk.id, "distance": item.distance,
                                 "similarity": retriever._similarity(item),
                                 "lexical_relevance": retriever._lexical_relevance(self.current_query, item)} for item in rows])
        original_retrieve = retriever.retrieve

        def retrieve(query, kb_ids):
            self.current_query = query
            return original_retrieve(query, kb_ids)

        retriever.retrieve = retrieve
        self.wrap(retriever, "retrieve", "retrieval", lambda args, kw: {"query": args[0], "kb_ids": list(args[1])})
        for method in ["decompose", "judge", "generate", "validate", "compose"]:
            self.wrap(agent._answering, method, method)


class ModelCallRecorder(logging.Handler):
    def __init__(self, observer):
        super().__init__()
        self.observer = observer

    def emit(self, record):
        if self.observer.record is not None:
            self.observer.record.setdefault("model_calls", []).append(json.loads(record.getMessage()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ids", default="")
    args = parser.parse_args()
    target = (ROOT / args.run_dir).resolve()
    if not target.is_relative_to(ROOT / "evaluation"):
        raise ValueError("Run directory must be inside evaluation")
    target.mkdir(parents=True, exist_ok=True)
    (target / "traces").mkdir(exist_ok=True)
    from validate_dataset import read_jsonl, validate, validate_live
    from rag_customer_service.config import Settings
    from rag_customer_service.bootstrap import build_container
    cases = read_jsonl(DATA / "regression-100.jsonl")
    chunks = read_jsonl(DATA / "chunks.snapshot.jsonl")
    corpus = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    validate(cases, chunks)
    validate_live(chunks, corpus)
    settings = Settings.from_env(ROOT / "rag-customer-service")
    safe_settings = {field.name: plain(getattr(settings, field.name)) for field in dataclasses.fields(settings)
                     if field.name not in {"qwen_api_key"}}
    identity = {"dataset_sha256": sha(DATA / "regression-100.jsonl"), "snapshot_sha256": sha(DATA / "chunks.snapshot.jsonl"),
                "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / "rag-customer-service/src").rglob("*.py"))},
                "settings": safe_settings, "temperature": 0, "rubric_sha256": sha(target / "rubric.md")}
    manifest_path = target / "run-manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["identity"] != identity:
            raise ValueError("Run configuration changed; use a new run directory")
    else:
        dump(manifest_path, {"started_at": datetime.now(timezone.utc).isoformat(), "identity": identity,
                             "entrypoint": "build_container(settings).agent.stream", "concurrency": 1,
                             "scorer": "Codex primary assistant; no Qwen self-scoring", "sdk_retries": "chat/embedding at most 1 retry; rerank at most 1 transient retry"})
    selected = set(args.ids.split(",")) if args.ids else {row["case_id"] for row in cases}
    observer = Observer()
    diagnostics = logging.getLogger("rag_customer_service.model_calls")
    diagnostics.setLevel(logging.INFO)
    diagnostics.addHandler(ModelCallRecorder(observer))
    diagnostics.propagate = False
    container = build_container(settings)
    observer.install(container)
    consecutive_errors = 0
    for row in cases:
        if row["case_id"] not in selected:
            continue
        path = target / "traces" / (row["case_id"] + ".json")
        if path.exists():
            print(f"SKIP {row['case_id']} already recorded", flush=True)
            continue
        started = time.perf_counter()
        record = {"case_id": row["case_id"], "question": row["question"], "history": row["history"],
                  "started_at": datetime.now(timezone.utc).isoformat(), "calls": [], "events": []}
        observer.record = record
        print(f"START {row['case_id']}", flush=True)
        for event in container.agent.stream(row["question"], row["history"], row["kb_ids"]):
            record["events"].append(plain(event))
        record["latency_ms"] = round((time.perf_counter() - started) * 1000)
        record["final_answer"] = "".join(event["message"] for event in record["events"] if event["type"] == "answer_delta")
        record["execution_status"] = "error" if any(e["type"] == "error" for e in record["events"]) else "completed"
        retrievals = [call for call in record["calls"] if call["stage"] == "retrieval" and "output" in call]
        ids = {f"{e['document_id']}:{e['chunk_index']}" for call in retrievals for e in call["output"]["evidences"]}
        record["strict_annotated_retrieval"] = coverage(row, ids)
        record["retrieved_chunk_ids"] = sorted(ids)
        record["retrieval_query_count"] = len(retrievals)
        record["chat_call_count"] = sum(call["stage"] == "chat" for call in record["calls"])
        record["embedding_call_count"] = sum(call["stage"] == "embedding" for call in record["calls"])
        dump(path, record)
        print(f"DONE {row['case_id']} status={record['execution_status']} ms={record['latency_ms']} chat={record['chat_call_count']}", flush=True)
        consecutive_errors = consecutive_errors + 1 if record["execution_status"] == "error" else 0
        if consecutive_errors >= 3:
            raise RuntimeError("Three consecutive service errors; inspect before continuing")


if __name__ == "__main__":
    main()
