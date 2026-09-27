from pathlib import Path

import pytest

from rag_customer_service.ingestion import DocumentIngestor
from rag_customer_service.knowledge_base import (
    KnowledgeBaseIngestionError,
    KnowledgeBaseManager,
)
from rag_customer_service.storage import DuplicateDocumentError, Storage


class FakeQwenClient:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.inputs: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.inputs.append(texts)
        if self.fail:
            raise RuntimeError("向量化失败")
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.chunks = {}

    def add_chunks(self, chunks, embeddings) -> None:
        self.chunks.update({chunk.id: chunk for chunk in chunks})
        if self.fail:
            raise RuntimeError("向量写入失败")

    def delete_document(self, document_id: str) -> None:
        self.chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self.chunks.items()
            if chunk.document_id != document_id
        }

    def delete_knowledge_base(self, kb_id: str) -> None:
        self.chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self.chunks.items()
            if chunk.kb_id != kb_id
        }


class FailingStorage:
    def __init__(self, storage: Storage, fail_stage: str) -> None:
        self._storage = storage
        self.fail_stage = fail_stage

    def __getattr__(self, name):
        return getattr(self._storage, name)

    def save_document_file(self, kb_id, document_id, content):
        path = self._storage.save_document_file(kb_id, document_id, content)
        if self.fail_stage == "file":
            raise RuntimeError("文件写入失败")
        return path

    def add_document(self, document):
        self._storage.add_document(document)
        if self.fail_stage == "sqlite":
            raise RuntimeError("元数据写入失败")


def make_manager(tmp_path, *, fail_stage: str | None = None):
    storage = Storage(tmp_path / "app.db", tmp_path / "uploads")
    manager_storage = (
        FailingStorage(storage, fail_stage) if fail_stage in {"file", "sqlite"} else storage
    )
    qwen = FakeQwenClient(fail=fail_stage == "embedding")
    vectors = FakeVectorStore(fail=fail_stage == "vector")
    manager = KnowledgeBaseManager(
        manager_storage,
        DocumentIngestor(chunk_size=50, chunk_overlap=10),
        qwen,
        vectors,
    )
    return manager, storage, qwen, vectors


def test_successful_ingestion_persists_all_layers_and_reports_progress(tmp_path):
    manager, storage, qwen, vectors = make_manager(tmp_path)
    kb = manager.create_knowledge_base("产品知识")
    progress = []

    document = manager.ingest_document(
        kb.id,
        "guide.md",
        ("# 指南\n\n" + "连接路由器。" * 20).encode("utf-8"),
        progress.append,
    )

    assert progress == ["校验", "切分", "向量化", "保存", "完成"]
    assert storage.list_documents(kb.id) == [document]
    assert document.file_path.read_text(encoding="utf-8").startswith("# 指南")
    assert document.chunk_count == len(vectors.chunks) == len(qwen.inputs[0])
    assert all(chunk.kb_id == kb.id for chunk in vectors.chunks.values())
    assert all(chunk.document_id == document.id for chunk in vectors.chunks.values())


def test_duplicate_content_is_rejected_before_second_embedding(tmp_path):
    manager, _, qwen, _ = make_manager(tmp_path)
    kb = manager.create_knowledge_base("产品知识")
    content = b"# Guide\n\nSame content"
    manager.ingest_document(kb.id, "first.md", content)

    with pytest.raises(DuplicateDocumentError):
        manager.ingest_document(kb.id, "second.md", content)

    assert len(qwen.inputs) == 1


@pytest.mark.parametrize("fail_stage", ["embedding", "file", "vector", "sqlite"])
def test_failed_ingestion_compensates_every_written_layer(tmp_path, fail_stage):
    manager, storage, _, vectors = make_manager(tmp_path, fail_stage=fail_stage)
    kb = manager.create_knowledge_base("产品知识")

    with pytest.raises(RuntimeError):
        manager.ingest_document(kb.id, "guide.md", b"# Guide\n\nContent")

    assert storage.list_documents(kb.id) == []
    assert list((storage.uploads_path / kb.id).glob("*.md")) == []
    assert vectors.chunks == {}


def test_cleanup_failure_keeps_original_error_as_cause(tmp_path, caplog):
    manager, _, _, vectors = make_manager(tmp_path, fail_stage="vector")
    kb = manager.create_knowledge_base("产品知识")

    def fail_cleanup(document_id):
        raise RuntimeError("清理失败")

    vectors.delete_document = fail_cleanup

    with pytest.raises(KnowledgeBaseIngestionError) as error:
        manager.ingest_document(kb.id, "guide.md", b"# Guide\n\nContent")

    assert str(error.value).startswith("文档处理失败")
    assert str(error.value.__cause__) == "向量写入失败"
    assert "Chroma 向量" in caplog.text


def test_document_and_knowledge_base_deletion_remove_only_target_data(tmp_path):
    manager, storage, _, vectors = make_manager(tmp_path)
    first_kb = manager.create_knowledge_base("产品知识")
    second_kb = manager.create_knowledge_base("售后知识")
    first = manager.ingest_document(first_kb.id, "one.md", b"First")
    retained = manager.ingest_document(first_kb.id, "two.md", b"Second")
    other = manager.ingest_document(second_kb.id, "three.md", b"Third")

    manager.delete_document(first_kb.id, first.id)

    assert storage.list_documents(first_kb.id) == [retained]
    assert not first.file_path.exists()
    assert retained.file_path.exists()
    assert {chunk.document_id for chunk in vectors.chunks.values()} == {
        retained.id,
        other.id,
    }

    manager.delete_knowledge_base(first_kb.id)

    assert storage.list_knowledge_bases() == [second_kb]
    assert other.file_path.exists()
    assert {chunk.document_id for chunk in vectors.chunks.values()} == {other.id}


def test_document_deletion_rejects_mismatched_knowledge_base_without_data_loss(tmp_path):
    manager, storage, _, vectors = make_manager(tmp_path)
    first_kb = manager.create_knowledge_base("产品知识")
    second_kb = manager.create_knowledge_base("售后知识")
    document = manager.ingest_document(second_kb.id, "warranty.md", b"Warranty")

    with pytest.raises(ValueError, match="文档不存在"):
        manager.delete_document(first_kb.id, document.id)

    assert storage.list_documents(second_kb.id) == [document]
    assert document.file_path.exists()
    assert {chunk.document_id for chunk in vectors.chunks.values()} == {document.id}


def test_six_product_documents_complete_local_lifecycle_without_network(tmp_path):
    manager, storage, qwen, _ = make_manager(tmp_path)
    knowledge_base = manager.create_knowledge_base("路由器产品知识")
    source_files = sorted((Path(__file__).parents[2] / "product data").glob("*.md"))

    documents = [
        manager.ingest_document(
            knowledge_base.id,
            source_file.name,
            source_file.read_bytes(),
        )
        for source_file in source_files
    ]

    assert len(source_files) == len(manager.list_documents(knowledge_base.id)) == 6
    assert len(qwen.inputs) == 6
    with pytest.raises(DuplicateDocumentError):
        manager.ingest_document(
            knowledge_base.id,
            source_files[0].name,
            source_files[0].read_bytes(),
        )

    manager.delete_document(knowledge_base.id, documents[0].id)

    assert not documents[0].file_path.exists()
    assert len(storage.list_documents(knowledge_base.id)) == 5
