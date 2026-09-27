from collections.abc import Sequence
from dataclasses import dataclass
import time

import httpx

from rag_customer_service.model_diagnostics import record_call


class RerankError(RuntimeError):
    """可安全展示的重排错误，不包含密钥或服务端原始响应。"""


@dataclass(frozen=True, slots=True)
class RerankResult:
    index: int
    score: float


class QwenReranker:
    def __init__(
        self,
        api_key: str,
        url: str,
        model: str = "qwen3-rerank",
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._url = url
        self._model = model
        self._transport = transport

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> tuple[RerankResult, ...]:
        if not documents:
            return ()
        count = min(top_n, len(documents))
        for attempt in (1, 2):
            started = time.perf_counter()
            try:
                with httpx.Client(transport=self._transport, timeout=httpx.Timeout(20.0, connect=30.0, write=5.0, pool=5.0)) as client:
                    response = client.post(
                        self._url,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json={"model": self._model, "query": query, "documents": list(documents), "top_n": count},
                    )
                    response.raise_for_status()
                break
            except (httpx.HTTPStatusError, httpx.RequestError) as error:
                record_call("rerank", self._model, started, attempt=attempt, error=error, secret=self._api_key)
                status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                if attempt == 1 and (status is None or status == 429 or status >= 500):
                    time.sleep(0.5)
                    continue
                if status is not None:
                    raise RerankError(f"千问重排失败（HTTP {status}），请检查权限、额度和接口配置") from None
                raise RerankError(f"千问重排失败，请检查网络连接或超时后重试（{type(error).__name__}）") from None

        try:
            payload = response.json()
            items = payload["results"]
            if not isinstance(items, list) or len(items) != count:
                raise ValueError
            results = []
            seen = set()
            for item in items:
                index, score = item["index"], item["relevance_score"]
                if (
                    type(index) is not int or not 0 <= index < len(documents)
                    or index in seen
                    or type(score) not in (int, float) or not 0 <= score <= 1
                ):
                    raise ValueError
                seen.add(index)
                results.append(RerankResult(index, float(score)))
            record_call("rerank", self._model, started, attempt=attempt, usage=payload.get("usage"),
                        request_id=payload.get("id") or response.headers.get("x-request-id"), secret=self._api_key)
            return tuple(sorted(results, key=lambda result: result.score, reverse=True))
        except (ValueError, KeyError, TypeError) as error:
            record_call("rerank", self._model, started, attempt=attempt, error=error, secret=self._api_key)
            raise RerankError("千问重排响应格式异常，请稍后重试") from None
