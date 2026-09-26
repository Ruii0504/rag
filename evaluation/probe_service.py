"""One diagnostic request; preserve provider error codes without exposing credentials."""
import json
from datetime import datetime, timezone

from run_regression import ROOT, dump
from rag_customer_service.config import Settings
from rag_customer_service.qwen import QwenClient

settings = Settings.from_env(ROOT / "rag-customer-service")
client = QwenClient(settings)
record = {"at": datetime.now(timezone.utc).isoformat(), "model": settings.qwen_chat_model}
try:
    response = client.chat_model.invoke([{"role": "user", "content": "Reply with OK."}])
    record.update(status="ok", response=str(response.content))
except Exception as error:
    record.update(status="error", error_type=type(error).__name__,
                  status_code=getattr(error, "status_code", None))
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error", body)
        if isinstance(nested, dict):
            record["provider_code"] = nested.get("code")
            record["provider_type"] = nested.get("type")
dump(ROOT / "evaluation/runs/2026-09-25-baseline/service-probe.json", record)
print(json.dumps(record, ensure_ascii=False))
