import json
import logging
import re
import time


logger = logging.getLogger("rag_customer_service.model_calls")


def record_call(stage, model, started, *, attempt=1, error=None, usage=None, request_id=None, secret=""):
    """仅记录允许的诊断字段，不记录提示词、正文、密钥或原始异常文本。"""
    record = {
        "stage": stage, "model": model, "attempt": attempt,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "status": "error" if error else "success",
    }
    if error:
        record["error_type"] = type(error).__name__
        response = getattr(error, "response", None)
        status = getattr(error, "status_code", None) or getattr(response, "status_code", None)
        if type(status) is int:
            record["http_status"] = status
        request_id = getattr(error, "request_id", None) or request_id
        if response is not None:
            request_id = response.headers.get("x-request-id") or request_id
    if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id) and not (secret and secret in request_id):
        record["request_id"] = request_id
    if isinstance(usage, dict):
        record["usage"] = {key: value for key, value in usage.items()
                           if key in {"input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"}
                           and type(value) is int and value >= 0}
    logger.log(logging.WARNING if error else logging.INFO, json.dumps(record, ensure_ascii=False))
