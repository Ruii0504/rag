"""Three tiny real rerank checks; no regression questions or KB documents sent."""
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_customer_service.config import Settings
from rag_customer_service.rerank import QwenReranker, RerankError

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "evaluation/runs/2026-09-26-rerank-timeout-fix/smoke.json"


class Recorder(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(json.loads(record.getMessage()))


def main():
    settings = Settings.from_env(ROOT / "rag-customer-service")
    recorder = Recorder()
    logger = logging.getLogger("rag_customer_service.model_calls")
    logger.setLevel(logging.INFO)
    logger.addHandler(recorder)
    logger.propagate = False
    reranker = QwenReranker(settings.qwen_api_key, settings.qwen_rerank_url, settings.qwen_rerank_model)
    rows = []
    for number in range(1, 4):
        started = time.perf_counter()
        offset = len(recorder.records)
        docs = ["USB port supports storage sharing, not printers.", "The router has six antennas."]
        if number == 2:
            docs.reverse()
        expected = 1 if number == 2 else 0
        row = {"probe": number, "started_at": datetime.now(timezone.utc).isoformat()}
        print(f"START probe={number}", flush=True)
        try:
            results = reranker.rerank("Can I connect a printer to the USB port?", docs, 2)
            row.update(status="success", passed=results[0].index == expected,
                       results=[{"index": item.index, "score": item.score} for item in results])
        except RerankError as error:
            row.update(status="error", passed=False, error=str(error))
        row.update(latency_ms=round((time.perf_counter() - started) * 1000),
                   attempts=recorder.records[offset:])
        rows.append(row)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": settings.qwen_rerank_model,
                   "rerank_source_sha256": hashlib.sha256((ROOT / "rag-customer-service/src/rag_customer_service/rerank.py").read_bytes()).hexdigest(),
                   "scope": "synthetic two-document rerank only; not 100-case regression", "probes": rows}
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"DONE probe={number} status={row['status']} passed={row['passed']} ms={row['latency_ms']}", flush=True)
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)


if __name__ == "__main__":
    main()
