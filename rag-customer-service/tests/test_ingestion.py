import hashlib

import pytest

from rag_customer_service.ingestion import DocumentIngestor, IngestionError


def test_prepare_normalizes_newlines_and_preserves_heading_paths():
    ingestor = DocumentIngestor(chunk_size=120, chunk_overlap=20)
    windows = "# 产品\r\n\r\n概览\r\n\r\n## 参数\r\n\r\n### 网络\r\n\r\nWiFi 7".encode(
        "utf-8"
    )
    unix = windows.replace(b"\r\n", b"\n")

    first = ingestor.prepare("guide.md", windows)
    second = ingestor.prepare("guide.md", unix)

    assert first.normalized_content == unix.decode("utf-8")
    assert first.content_hash == second.content_hash
    assert first.content_hash == hashlib.sha256(unix).hexdigest()
    assert [chunk.chunk_index for chunk in first.chunks] == list(
        range(len(first.chunks))
    )
    assert all(chunk.document_name == "guide.md" for chunk in first.chunks)
    assert [chunk.heading_path for chunk in first.chunks] == [
        ("产品",),
        ("产品", "参数", "网络"),
    ]


def test_long_section_is_recursively_split_and_reproducible():
    ingestor = DocumentIngestor(chunk_size=60, chunk_overlap=10)
    content = ("# 故障排查\n\n" + "请检查网络连接。" * 20).encode("utf-8")

    first = ingestor.prepare("troubleshooting.md", content)
    second = ingestor.prepare("troubleshooting.md", content)

    assert len(first.chunks) > 1
    assert first.chunks == second.chunks
    assert all(chunk.heading_path == ("故障排查",) for chunk in first.chunks)
    assert all(0 < len(chunk.content) <= 60 for chunk in first.chunks)


def test_document_without_headings_gets_empty_heading_path():
    prepared = DocumentIngestor(chunk_size=100, chunk_overlap=10).prepare(
        "notes.md",
        "这是一段没有标题的正文。".encode("utf-8"),
    )

    assert len(prepared.chunks) == 1
    assert prepared.chunks[0].heading_path == ()
    assert prepared.chunks[0].content == "这是一段没有标题的正文。"


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("manual.txt", b"content", "Markdown"),
        ("manual.md", b" \r\n\t", "空"),
        ("manual.md", b"\xff\xfe", "UTF-8"),
    ],
)
def test_invalid_documents_are_rejected(filename, content, message):
    ingestor = DocumentIngestor(chunk_size=100, chunk_overlap=10)

    with pytest.raises(IngestionError, match=message):
        ingestor.prepare(filename, content)


def test_content_change_changes_hash():
    ingestor = DocumentIngestor(chunk_size=100, chunk_overlap=10)

    first = ingestor.prepare("guide.md", "# 指南\n\n内容 A".encode("utf-8"))
    second = ingestor.prepare("guide.md", "# 指南\n\n内容 B".encode("utf-8"))

    assert first.content_hash != second.content_hash
