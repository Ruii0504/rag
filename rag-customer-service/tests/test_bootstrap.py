import inspect
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_customer_service.agent import CustomerServiceAgent
from rag_customer_service.bootstrap import AppContainer, build_container, get_container
from rag_customer_service.config import ConfigurationError, Settings
from rag_customer_service.knowledge_base import KnowledgeBaseManager


def test_bootstrap_has_no_streamlit_runtime_dependency():
    import rag_customer_service.bootstrap as bootstrap

    assert "streamlit" not in inspect.getsource(bootstrap)


class FakeQwenClient:
    def __init__(self, settings):
        self.settings = settings

    def chat(self, messages):
        return messages[-1]["content"]

    def stream_chat(self, messages):
        yield "回答"

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def make_settings(tmp_path):
    return Settings(
        qwen_api_key="test-key",
        qwen_base_url="https://example.test/v1",
        qwen_chat_model="chat-test",
        qwen_embedding_model="embedding-test",
        qwen_embedding_dimension=2,
        sqlite_path=tmp_path / "data" / "app.db",
        chroma_path=tmp_path / "data" / "chroma",
        uploads_path=tmp_path / "data" / "uploads",
        chunk_size=100,
        chunk_overlap=10,
        retrieval_top_k=3,
        retrieval_score_threshold=0.45,
    )


def test_build_container_exposes_only_page_services_and_initializes_data(tmp_path):
    settings = make_settings(tmp_path)

    container = build_container(settings, qwen_factory=FakeQwenClient)

    assert isinstance(container, AppContainer)
    assert set(container.__dataclass_fields__) == {"knowledge_bases", "agent"}
    assert isinstance(container.knowledge_bases, KnowledgeBaseManager)
    assert isinstance(container.agent, CustomerServiceAgent)
    assert settings.sqlite_path.exists()
    assert settings.chroma_path.is_dir()


def test_get_container_reuses_same_resources_across_reruns(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(lambda cls, base_dir=None: settings),
    )
    get_container.cache_clear()

    first = get_container(Path(tmp_path))
    second = get_container(Path(tmp_path))

    assert first is second
    get_container.cache_clear()


def test_invalid_configuration_creates_no_partial_data(tmp_path, monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    get_container.cache_clear()

    with pytest.raises(ConfigurationError):
        get_container(Path(tmp_path))

    assert not (tmp_path / "data").exists()
    get_container.cache_clear()


@pytest.mark.parametrize("rerank_status", [200, 401])
def test_production_api_uses_rerank_and_keeps_semantic_gate(tmp_path, monkeypatch, rerank_status):
    from rag_customer_service.api import create_app

    settings = replace(make_settings(tmp_path), qwen_rerank_url="https://example.test/reranks")

    class ModelClient(FakeQwenClient):
        def chat(self, messages):
            if "standalone_question" in messages[0]["content"]:
                return json.dumps({"standalone_question": "价格？", "subquestions": ["价格？"]})
            if "证据判定器" in messages[0]["content"]:
                return json.dumps({"decisions": [{"subquestion_id": "q1", "answerable": False,
                    "evidence_ids": [], "missing_information": "价格", "reason": "无价格证据"}]})
            raise AssertionError("No answer generation is permitted for unsupported evidence")

    requests = []

    def respond(transport, request):
        requests.append(request)
        assert request.url == "https://example.test/reranks"
        assert json.loads(request.content)["model"] == "qwen3-rerank"
        return httpx.Response(rerank_status, json={"results": [{"index": 0, "relevance_score": 0.99}]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", respond)
    container = build_container(settings, qwen_factory=ModelClient)
    kb = container.knowledge_bases.create_knowledge_base("Product")
    document = container.knowledge_bases.ingest_document(kb.id, "product.md", b"# Product\nSupports WiFi.")
    with TestClient(create_app(container, settings)) as client:
        status = client.get("/api/status").json()
        response = client.post("/api/chat/stream", json={"question": "价格？", "kb_ids": [kb.id], "history": []})
        assert response.status_code == 200

    assert len(requests) == 1
    assert status["rerank_model"] == "qwen3-rerank"
    assert status["retrieval_top_k"] == 3
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    if rerank_status == 401:
        assert events[-1]["type"] == "error"
        assert events[-1]["stage"] == "rerank"
        assert "HTTP 401" in events[-1]["message"]
        assert all(event["type"] not in ("answer_delta", "completed") for event in events)
    else:
        evidence_event = next(event for event in events if event["type"] == "evidence")
        evidence = evidence_event["payload"]["result"]["evidences"][0]
        assert evidence["document_id"] == document.id
        assert evidence["rerank_score"] == 0.99
        assert evidence["similarity"] == 1.0
        assert evidence["support_status"] == "related"
        assert any(event["stage"] == "refuse" for event in events)
        assert events[-1]["type"] == "completed"
