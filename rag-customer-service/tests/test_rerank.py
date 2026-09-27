import json

import httpx
import pytest

from rag_customer_service.rerank import QwenReranker, RerankError


def test_rerank_uses_flat_request_and_preserves_original_document_indices():
    def respond(request):
        assert request.url == "https://example.test/reranks"
        assert request.headers["authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "model": "qwen3-rerank",
            "query": "USB 能连接打印机吗？",
            "documents": ["支持 USB 存储共享", "USB 不支持打印机"],
            "top_n": 2,
        }
        return httpx.Response(200, json={"results": [
            {"index": 1, "relevance_score": 0.98},
            {"index": 0, "relevance_score": 0.2},
        ]})

    reranker = QwenReranker("test-key", "https://example.test/reranks", transport=httpx.MockTransport(respond))
    result = reranker.rerank("USB 能连接打印机吗？", ["支持 USB 存储共享", "USB 不支持打印机"], 7)

    assert [(item.index, item.score) for item in result] == [(1, 0.98), (0, 0.2)]


@pytest.mark.parametrize("payload", [
    {"results": [{"index": -1, "relevance_score": 0.9}]},
    {"results": [{"index": 2, "relevance_score": 0.9}]},
    {"results": [{"index": True, "relevance_score": 0.9}]},
    {"results": [{"index": 0, "relevance_score": "0.9"}]},
    {"results": [{"index": 0, "relevance_score": 1.1}]},
    {"results": []},
    {"results": [{"index": 0}]},
    {"error": "test-key must not be exposed"},
    None,
])
def test_invalid_rerank_response_fails_explicitly_without_inventing_results(payload):
    reranker = QwenReranker("test-key", "https://example.test/reranks", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=payload)
    ))
    with pytest.raises(RerankError, match="响应格式异常") as error:
        reranker.rerank("question", ["one", "two"], 1)
    assert "test-key" not in str(error.value)


def test_duplicate_indices_are_rejected():
    reranker = QwenReranker("test-key", "https://example.test/reranks", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"results": [
            {"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.8},
        ]})
    ))
    with pytest.raises(RerankError, match="响应格式异常"):
        reranker.rerank("question", ["one", "two"], 2)


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_failure_is_safe_and_does_not_fall_back(status):
    reranker = QwenReranker("test-key", "https://example.test/reranks", transport=httpx.MockTransport(
        lambda request: httpx.Response(status, text="test-key")
    ))
    with pytest.raises(RerankError, match=f"HTTP {status}") as error:
        reranker.rerank("question", ["one"], 1)
    assert "test-key" not in str(error.value)


def test_timeout_is_explicit_and_empty_documents_make_no_request():
    def timeout(request):
        raise httpx.ReadTimeout("test-key", request=request)

    reranker = QwenReranker("test-key", "https://example.test/reranks", transport=httpx.MockTransport(timeout))
    assert reranker.rerank("question", [], 7) == ()
    with pytest.raises(RerankError, match="网络连接或超时"):
        reranker.rerank("question", ["one"], 1)


@pytest.mark.parametrize("failure", ["connection", 429, 503])
def test_transient_failure_retries_once_and_records_safe_diagnostics(failure, caplog):
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            if failure == "connection":
                raise httpx.ConnectError("test-key private document", request=request)
            return httpx.Response(failure, text="test-key private document")
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}], "usage": {"total_tokens": 12}})

    caplog.set_level("INFO", logger="rag_customer_service.model_calls")
    result = QwenReranker("test-key", "https://example.test/reranks", transport=httpx.MockTransport(respond)).rerank("private question", ["private document"], 1)
    assert result[0].score == 0.9
    assert len(calls) == 2
    assert '"attempt": 2' in caplog.text
    assert '"total_tokens": 12' in caplog.text
    assert "test-key" not in caplog.text
    assert "private document" not in caplog.text
    assert "private question" not in caplog.text


def test_authentication_failure_is_not_retried(caplog):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(401, headers={"x-request-id": "safe-request-123"}, text="test-key")

    caplog.set_level("INFO", logger="rag_customer_service.model_calls")
    with pytest.raises(RerankError, match="HTTP 401"):
        QwenReranker("test-key", "https://example.test/reranks", transport=httpx.MockTransport(respond)).rerank("question", ["one"], 1)
    assert len(calls) == 1
    assert "safe-request-123" in caplog.text
    assert "test-key" not in caplog.text


def test_slow_tls_handshake_within_thirty_seconds_can_complete():
    def respond(request):
        # Emulate an external TLS handshake that needs eight seconds without sleeping.
        if request.extensions["timeout"]["connect"] < 8.0:
            raise httpx.ConnectTimeout("TLS handshake timed out", request=request)
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.95}]})

    reranker = QwenReranker("test-key", "https://example.test/reranks", transport=httpx.MockTransport(respond))
    result = reranker.rerank("question", ["evidence"], 1)
    assert [(item.index, item.score) for item in result] == [(0, 0.95)]


def test_persistent_connect_timeout_is_bounded_and_never_invents_results():
    calls = []

    def respond(request):
        calls.append(request)
        connect_timeout = request.extensions["timeout"]["connect"]
        assert connect_timeout is not None and 0 < connect_timeout <= 30.0
        raise httpx.ConnectTimeout("test-key handshake timeout", request=request)

    reranker = QwenReranker("test-key", "https://example.test/reranks", transport=httpx.MockTransport(respond))
    with pytest.raises(RerankError, match="ConnectTimeout") as error:
        reranker.rerank("question", ["evidence"], 1)
    assert len(calls) == 2
    assert "test-key" not in str(error.value)
