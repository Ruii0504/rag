from pathlib import Path

import pytest

from rag_customer_service.config import ConfigurationError, Settings


def test_settings_reads_environment_and_resolves_data_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_API_KEY", "test-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("QWEN_CHAT_MODEL", "qwen-test")
    monkeypatch.setenv("QWEN_EMBEDDING_MODEL", "embedding-test")
    monkeypatch.setenv("QWEN_EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv("SQLITE_PATH", "runtime/app.db")
    monkeypatch.setenv("CHROMA_PATH", "runtime/chroma")
    monkeypatch.setenv("UPLOADS_PATH", "runtime/uploads")
    monkeypatch.setenv("CHUNK_SIZE", "600")
    monkeypatch.setenv("CHUNK_OVERLAP", "60")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "4")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.55")

    settings = Settings.from_env(base_dir=tmp_path)

    assert settings.qwen_api_key == "test-key"
    assert settings.qwen_base_url == "https://example.test/v1"
    assert settings.qwen_chat_model == "qwen-test"
    assert settings.qwen_embedding_model == "embedding-test"
    assert settings.qwen_embedding_dimension == 768
    assert settings.sqlite_path == tmp_path / "runtime/app.db"
    assert settings.chroma_path == tmp_path / "runtime/chroma"
    assert settings.uploads_path == tmp_path / "runtime/uploads"
    assert settings.chunk_size == 600
    assert settings.chunk_overlap == 60
    assert settings.retrieval_top_k == 4
    assert settings.retrieval_score_threshold == pytest.approx(0.55)


def test_settings_rejects_missing_qwen_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="QWEN_API_KEY"):
        Settings.from_env(base_dir=tmp_path)


def test_settings_rejects_overlap_not_smaller_than_chunk_size(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    with pytest.raises(ConfigurationError, match="CHUNK_OVERLAP"):
        Settings.from_env(base_dir=Path(tmp_path))


def test_settings_uses_confirmed_candidate_retrieval_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_API_KEY", "test-key")
    monkeypatch.delenv("RETRIEVAL_TOP_K", raising=False)
    monkeypatch.delenv("RETRIEVAL_SCORE_THRESHOLD", raising=False)

    settings = Settings.from_env(base_dir=tmp_path)

    assert settings.retrieval_top_k == 7
    assert settings.retrieval_score_threshold == pytest.approx(0.5)


def test_rerank_has_independent_endpoint_without_changing_chat_or_embeddings(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_API_KEY", "test-key")
    monkeypatch.setenv("QWEN_CHAT_MODEL", "qwen3.7-plus")
    monkeypatch.setenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.delenv("QWEN_RERANK_URL", raising=False)
    settings = Settings.from_env(tmp_path)

    assert settings.qwen_rerank_model == "qwen3-rerank"
    assert settings.qwen_rerank_url == "https://maas.qianwenaiapi.com/compatible-api/v1/reranks"
    assert settings.qwen_chat_model == "qwen3.7-plus"
    assert settings.qwen_embedding_model == "text-embedding-v4"

    monkeypatch.setenv("QWEN_RERANK_URL", "https://example.test/reranks")
    assert Settings.from_env(tmp_path).qwen_rerank_url == "https://example.test/reranks"
