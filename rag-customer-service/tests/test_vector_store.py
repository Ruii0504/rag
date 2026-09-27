from pathlib import Path

import pytest

from rag_customer_service.models import Chunk
from rag_customer_service.vector_store import (
    ChromaVectorStore,
    VectorStoreConfigurationError,
)


def make_chunk(
    chunk_id: str,
    kb_id: str,
    document_id: str,
    content: str,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        kb_id=kb_id,
        document_id=document_id,
        document_name=f"{document_id}.md",
        heading_path=("产品", "参数"),
        chunk_index=0,
        content=content,
        content_hash=f"hash-{chunk_id}",
    )


def make_store(path: Path, model: str = "embedding-a", dimension: int = 2):
    return ChromaVectorStore(path, model, dimension)


def test_search_filters_multiple_knowledge_bases_and_restores_matches(tmp_path):
    store = make_store(tmp_path / "chroma")
    chunks = [
        make_chunk("c-1", "kb-1", "doc-1", "第一台路由器"),
        make_chunk("c-2", "kb-2", "doc-2", "第二台路由器"),
        make_chunk("c-3", "kb-3", "doc-3", "不应返回"),
    ]
    store.add_chunks(chunks, [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    matches = store.search([1.0, 0.0], ["kb-1", "kb-2"], top_k=5)

    assert [match.chunk.id for match in matches] == ["c-1", "c-2"]
    assert matches[0].distance == pytest.approx(0.0)
    assert matches[1].distance == pytest.approx(2.0)
    assert matches[0].chunk.document_name == "doc-1.md"
    assert matches[0].chunk.heading_path == ("产品", "参数")
    assert matches[0].chunk.content == "第一台路由器"


def test_delete_document_and_knowledge_base_are_scoped(tmp_path):
    store = make_store(tmp_path / "chroma")
    store.add_chunks(
        [
            make_chunk("c-1", "kb-1", "doc-1", "文档一"),
            make_chunk("c-2", "kb-1", "doc-2", "文档二"),
            make_chunk("c-3", "kb-2", "doc-3", "文档三"),
        ],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )

    store.delete_document("doc-1")

    assert [
        match.chunk.id
        for match in store.search([1.0, 0.0], ["kb-1", "kb-2"], top_k=10)
    ] == ["c-2", "c-3"]

    store.delete_knowledge_base("kb-1")

    assert [
        match.chunk.id
        for match in store.search([1.0, 0.0], ["kb-1", "kb-2"], top_k=10)
    ] == ["c-3"]


@pytest.mark.parametrize(
    ("model", "dimension"),
    [("embedding-b", 2), ("embedding-a", 3)],
)
def test_existing_collection_rejects_model_or_dimension_change(
    tmp_path,
    model,
    dimension,
):
    path = tmp_path / "chroma"
    make_store(path)

    with pytest.raises(VectorStoreConfigurationError, match="重新向量化"):
        make_store(path, model=model, dimension=dimension)
