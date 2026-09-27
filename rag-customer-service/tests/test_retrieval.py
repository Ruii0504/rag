import pytest

from rag_customer_service.models import Chunk, VectorMatch
from rag_customer_service.retrieval import Retriever


def make_match(
    chunk_id: str,
    kb_id: str,
    distance: float,
    *,
    heading_path: tuple[str, ...] = ("指南",),
    content: str | None = None,
) -> VectorMatch:
    return VectorMatch(
        chunk=Chunk(
            id=chunk_id,
            kb_id=kb_id,
            document_id=f"doc-{chunk_id}",
            document_name=f"{chunk_id}.md",
            heading_path=heading_path,
            chunk_index=0,
            content=content or f"证据 {chunk_id}",
            content_hash=f"hash-{chunk_id}",
        ),
        distance=distance,
    )


class FakeQwenClient:
    def __init__(self) -> None:
        self.queries = []

    def embed_query(self, query):
        self.queries.append(query)
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self, matches=(), *, honor_top_k: bool = False) -> None:
        self.matches = tuple(matches)
        self.honor_top_k = honor_top_k
        self.calls = []

    def search(self, embedding, kb_ids, top_k):
        self.calls.append((embedding, list(kb_ids), top_k))
        matches = tuple(match for match in self.matches if match.chunk.kb_id in kb_ids)
        return matches[:top_k] if self.honor_top_k else matches


def test_no_selected_knowledge_base_returns_explainable_empty_result():
    qwen = FakeQwenClient()
    vectors = FakeVectorStore()
    retriever = Retriever(qwen, vectors, top_k=3, score_threshold=0.45)

    result = retriever.retrieve("如何重置？", [])

    assert result.query == "如何重置？"
    assert result.evidences == ()
    assert result.has_sufficient_evidence is False
    assert result.reason == "未选择知识库"
    assert qwen.queries == []
    assert vectors.calls == []


@pytest.mark.parametrize("selected", [["kb-1"], ["kb-1", "kb-2"]])
def test_selected_scope_is_forwarded_without_leaking_other_knowledge_bases(
    selected,
):
    qwen = FakeQwenClient()
    vectors = FakeVectorStore(
        [
            make_match("one", "kb-1", 0.0),
            make_match("two", "kb-2", 0.1),
            make_match("three", "kb-3", 0.0),
        ]
    )
    retriever = Retriever(qwen, vectors, top_k=5, score_threshold=0.0)

    result = retriever.retrieve("如何重置？", selected)

    assert qwen.queries == ["如何重置？"]
    assert vectors.calls == [([1.0, 0.0], selected, 100)]
    assert {evidence.kb_id for evidence in result.evidences} == set(selected)
    assert "kb-3" not in {evidence.kb_id for evidence in result.evidences}


def test_results_are_sorted_truncated_filtered_and_numbered():
    matches = [
        make_match("middle", "kb-1", 0.5),
        make_match("best", "kb-1", 0.0),
        make_match("weak", "kb-1", 4.0),
        make_match("second", "kb-1", 0.25),
    ]
    retriever = Retriever(
        FakeQwenClient(),
        FakeVectorStore(matches),
        top_k=3,
        score_threshold=0.7,
    )

    result = retriever.retrieve("问题", ["kb-1"])

    assert [evidence.document_name for evidence in result.evidences] == [
        "best.md",
        "second.md",
    ]
    assert [evidence.reference_id for evidence in result.evidences] == [1, 2]
    assert [evidence.similarity for evidence in result.evidences] == pytest.approx(
        [1.0, 0.8]
    )
    assert [evidence.distance for evidence in result.evidences] == [0.0, 0.25]
    assert result.has_sufficient_evidence is True
    assert result.reason == ""


def test_equal_threshold_is_kept_but_all_lower_results_are_insufficient():
    at_threshold = Retriever(
        FakeQwenClient(),
        FakeVectorStore([make_match("equal", "kb-1", 1.0)]),
        top_k=1,
        score_threshold=0.5,
    ).retrieve("问题", ["kb-1"])
    insufficient = Retriever(
        FakeQwenClient(),
        FakeVectorStore([make_match("weak", "kb-1", 1.01)]),
        top_k=1,
        score_threshold=0.5,
    ).retrieve("问题", ["kb-1"])

    assert at_threshold.has_sufficient_evidence is True
    assert at_threshold.evidences[0].similarity == pytest.approx(0.5)
    assert insufficient.has_sufficient_evidence is False
    assert insufficient.evidences == ()
    assert insufficient.reason == "未检索到达到相似度阈值的证据"


def test_confirmed_candidate_policy_keeps_at_most_seven_results_above_half():
    matches = [
        make_match(f"candidate-{index}", "kb-1", 0.5 + index * 0.01)
        for index in range(8)
    ]
    vectors = FakeVectorStore(matches)
    retriever = Retriever(
        FakeQwenClient(),
        vectors,
        top_k=7,
        score_threshold=0.5,
    )

    result = retriever.retrieve("价格是多少？", ["kb-1"])

    assert len(result.evidences) == 7
    assert all(evidence.similarity >= 0.5 for evidence in result.evidences)
    assert vectors.calls == [([1.0, 0.0], ["kb-1"], 140)]


def test_heading_match_reranks_direct_size_evidence_from_broader_internal_pool():
    irrelevant = [
        make_match(
            f"general-{index}",
            "kb-1",
            0.1 + index * 0.01,
            heading_path=("星云智联 AX6000 智能路由器", "常见问题"),
            content="星云智联 AX6000 支持家庭网络功能。",
        )
        for index in range(7)
    ]
    direct = make_match(
        "dimensions",
        "kb-1",
        0.8,
        heading_path=("星云智联 AX6000 智能路由器", "尺寸、材质与散热"),
        content="整机含天线尺寸约为长 250 毫米、宽 150 毫米、高 200 毫米。",
    )
    vectors = FakeVectorStore([*irrelevant, direct], honor_top_k=True)
    retriever = Retriever(
        FakeQwenClient(),
        vectors,
        top_k=7,
        score_threshold=0.5,
    )

    result = retriever.retrieve(
        "星云智联 AX6000 智能路由器的尺寸是多少？",
        ["kb-1"],
    )

    assert len(result.evidences) == 7
    assert result.evidences[0].document_name == "dimensions.md"
    assert result.evidences[0].similarity == pytest.approx(1 / 1.8)
