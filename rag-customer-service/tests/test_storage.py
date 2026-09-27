from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_customer_service.models import Document, DocumentStatus
from rag_customer_service.storage import (
    DuplicateDocumentError,
    DuplicateKnowledgeBaseError,
    Storage,
)


def make_storage(tmp_path):
    return Storage(tmp_path / "app.db", tmp_path / "uploads")


def make_document(kb_id: str, document_id: str, content_hash: str) -> Document:
    return Document(
        id=document_id,
        kb_id=kb_id,
        filename="../../../不可信文件名.md",
        content_hash=content_hash,
        file_path=Path("uploads") / f"{document_id}.md",
        chunk_count=2,
        status=DocumentStatus.READY,
        created_at=datetime(2026, 8, 26, tzinfo=UTC),
    )


def test_knowledge_base_lifecycle_and_unique_name(tmp_path):
    storage = make_storage(tmp_path)

    created = storage.create_knowledge_base("产品知识")

    assert storage.list_knowledge_bases() == [created]
    with pytest.raises(DuplicateKnowledgeBaseError):
        storage.create_knowledge_base("产品知识")

    storage.delete_knowledge_base(created.id)
    assert storage.list_knowledge_bases() == []


def test_document_hash_is_unique_only_within_same_knowledge_base(tmp_path):
    storage = make_storage(tmp_path)
    first_kb = storage.create_knowledge_base("产品知识")
    second_kb = storage.create_knowledge_base("售后知识")
    first = make_document(first_kb.id, "doc-1", "same-hash")
    second = make_document(second_kb.id, "doc-2", "same-hash")

    storage.add_document(first)
    with pytest.raises(DuplicateDocumentError):
        storage.add_document(make_document(first_kb.id, "doc-3", "same-hash"))
    storage.add_document(second)

    assert storage.list_documents(first_kb.id) == [first]
    assert storage.list_documents(second_kb.id) == [second]


def test_document_file_path_uses_ids_and_deletion_is_explicit(tmp_path):
    storage = make_storage(tmp_path)
    kb = storage.create_knowledge_base("产品知识")

    saved_path = storage.save_document_file(kb.id, "doc-1", "# 内容")

    assert saved_path == tmp_path / "uploads" / kb.id / "doc-1.md"
    assert saved_path.read_text(encoding="utf-8") == "# 内容"
    assert "不可信文件名" not in str(saved_path)

    storage.delete_document_file(kb.id, "doc-1")
    assert not saved_path.exists()


def test_deleting_document_metadata_does_not_remove_other_documents(tmp_path):
    storage = make_storage(tmp_path)
    kb = storage.create_knowledge_base("产品知识")
    first = make_document(kb.id, "doc-1", "hash-1")
    second = make_document(kb.id, "doc-2", "hash-2")
    storage.add_document(first)
    storage.add_document(second)

    storage.delete_document(kb.id, first.id)

    assert storage.list_documents(kb.id) == [second]
