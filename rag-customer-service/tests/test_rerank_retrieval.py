import json

import httpx
import pytest

from rag_customer_service.models import Chunk
from rag_customer_service.rerank import QwenReranker
from rag_customer_service.retrieval import Retriever
from rag_customer_service.vector_store import ChromaVectorStore


class QueryEmbeddings:
    def embed_query(self, query):
        return [1.0, 0.0]


def test_rerank_promotes_low_vector_score_and_preserves_sources_and_kb_scope(tmp_path):
    vectors = ChromaVectorStore(tmp_path / "chroma", "embedding-test", 2)
    chunks = [
        Chunk(f"chunk-{index}", "kb-selected" if index < 8 else "kb-private",
              f"doc-{index}", f"file-{index}.md", ("Product", f"Section {index}"),
              index, "USB 不支持打印机" if index == 7 else f"General info {index}", f"hash-{index}")
        for index in range(9)
    ]
    vectors.add_chunks(chunks, [[1.0 + index * 0.01, 0.0] for index in range(7)] + [[2.1, 0.0], [1.0, 0.0]])

    def respond(request):
        body = json.loads(request.content)
        assert body["top_n"] == 7
        assert len(body["documents"]) == 8
        assert "Section 7" in body["documents"][7]
        assert "USB 不支持打印机" in body["documents"][7]
        assert all("file-8.md" not in text for text in body["documents"])
        return httpx.Response(200, json={"results": [
            {"index": index, "relevance_score": score}
            for index, score in [(7, 0.98), (0, 0.8), (1, 0.7), (2, 0.6), (3, 0.4), (4, 0.3), (5, 0.2)]
        ]})

    retriever = Retriever(QueryEmbeddings(), vectors, 7, 0.5,
                          reranker=QwenReranker("test-key", "https://example.test/reranks", transport=httpx.MockTransport(respond)))
    result = retriever.retrieve("USB 能连接打印机吗？", ["kb-selected"])

    assert [e.document_id for e in result.evidences] == ["doc-7", "doc-0", "doc-1", "doc-2", "doc-3", "doc-4", "doc-5"]
    assert [e.reference_id for e in result.evidences] == list(range(1, 8))
    first = result.evidences[0]
    assert first.chunk_index == 7
    assert first.content == "USB 不支持打印机"
    assert first.heading_path == ("Product", "Section 7")
    assert first.similarity < 0.5
    assert first.rerank_score == pytest.approx(0.98)
    assert {e.kb_id for e in result.evidences} == {"kb-selected"}
