from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_customer_service.config import Settings
from rag_customer_service.qwen import QwenClient, QwenClientError


def make_settings() -> Settings:
    return Settings(
        qwen_api_key="secret-key",
        qwen_base_url="https://example.test/v1",
        qwen_chat_model="qwen-test",
        qwen_embedding_model="embedding-test",
        qwen_embedding_dimension=768,
        sqlite_path=Path("data/app.db"),
        chroma_path=Path("data/chroma"),
        uploads_path=Path("data/uploads"),
        chunk_size=800,
        chunk_overlap=100,
        retrieval_top_k=5,
        retrieval_score_threshold=0.45,
    )


class FakeChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return SimpleNamespace(content="完整回答")

    def stream(self, messages):
        self.invocations.append(messages)
        yield SimpleNamespace(content="流式")
        yield SimpleNamespace(content="回答")


class FakeEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.document_inputs = []
        self.query_inputs = []

    def embed_documents(self, texts):
        self.document_inputs.append(texts)
        return [[1.0, 0.0], [0.0, 1.0]]

    def embed_query(self, text):
        self.query_inputs.append(text)
        return [0.5, 0.5]


class BatchLimitedEmbeddings(FakeEmbeddings):
    def embed_documents(self, texts):
        if len(texts) > 10:
            raise ValueError("批量不能超过 10 条")
        self.document_inputs.append(texts)
        return [[float(text.split()[1])] for text in texts]


def test_qwen_client_passes_connection_and_model_settings():
    client = QwenClient(
        make_settings(),
        chat_factory=FakeChatModel,
        embeddings_factory=FakeEmbeddings,
    )

    assert client.chat_model.kwargs == {
        "api_key": "secret-key",
        "base_url": "https://example.test/v1",
        "model": "qwen-test",
        "temperature": 0,
        "timeout": 30.0,
        "max_retries": 1,
    }
    assert client.embeddings.kwargs == {
        "api_key": "secret-key",
        "base_url": "https://example.test/v1",
        "model": "embedding-test",
        "dimensions": 768,
        "check_embedding_ctx_length": False,
        "request_timeout": 30.0,
        "max_retries": 1,
    }


def test_chat_and_stream_chat_preserve_messages_and_chunk_order():
    client = QwenClient(
        make_settings(),
        chat_factory=FakeChatModel,
        embeddings_factory=FakeEmbeddings,
    )
    messages = [{"role": "user", "content": "你好"}]

    assert client.chat(messages) == "完整回答"
    assert list(client.stream_chat(messages)) == ["流式", "回答"]
    assert client.chat_model.invocations == [messages, messages]


def test_embedding_methods_keep_all_inputs():
    client = QwenClient(
        make_settings(),
        chat_factory=FakeChatModel,
        embeddings_factory=FakeEmbeddings,
    )

    assert client.embed_documents(["第一段", "第二段"]) == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert client.embed_query("问题") == [0.5, 0.5]
    assert client.embeddings.document_inputs == [["第一段", "第二段"]]
    assert client.embeddings.query_inputs == ["问题"]


def test_document_embeddings_are_submitted_in_batches_of_at_most_ten():
    client = QwenClient(
        make_settings(),
        chat_factory=FakeChatModel,
        embeddings_factory=BatchLimitedEmbeddings,
    )
    texts = [f"第 {index} 段" for index in range(19)]

    embeddings = client.embed_documents(texts)

    assert embeddings == [[float(index)] for index in range(19)]
    assert [len(batch) for batch in client.embeddings.document_inputs] == [10, 9]


def test_adapter_error_does_not_expose_api_key():
    class FailingChatModel(FakeChatModel):
        def invoke(self, messages):
            raise RuntimeError("request failed with secret-key")

    client = QwenClient(
        make_settings(),
        chat_factory=FailingChatModel,
        embeddings_factory=FakeEmbeddings,
    )

    with pytest.raises(QwenClientError) as error:
        client.chat([{"role": "user", "content": "你好"}])

    assert "secret-key" not in str(error.value)


def test_chat_diagnostics_record_usage_and_error_type_without_content(caplog):
    class ObservedChat(FakeChatModel):
        def invoke(self, messages):
            if messages[0]["content"] == "fail":
                raise RuntimeError("secret-key private-text")
            return SimpleNamespace(content="private-text", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    caplog.set_level("INFO", logger="rag_customer_service.model_calls")
    client = QwenClient(make_settings(), chat_factory=ObservedChat, embeddings_factory=FakeEmbeddings)
    assert client.chat([{"role": "user", "content": "private-text"}]) == "private-text"
    with pytest.raises(QwenClientError):
        client.chat([{"role": "user", "content": "fail"}])
    assert '"total_tokens": 15' in caplog.text
    assert "RuntimeError" in caplog.text
    assert "secret-key" not in caplog.text
    assert "private-text" not in caplog.text
