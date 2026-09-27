import hashlib
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_customer_service.models import PreparedChunk, PreparedDocument


class IngestionError(ValueError):
    """文档摄取前置校验失败。"""


class DocumentIngestor:
    _heading_pattern = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            keep_separator=True,
        )

    def prepare(self, filename: str, content: bytes) -> PreparedDocument:
        if Path(filename).suffix.lower() != ".md":
            raise IngestionError("仅支持 Markdown (.md) 文件")

        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError:
            raise IngestionError("文件必须使用 UTF-8 编码") from None

        normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.strip():
            raise IngestionError("文件内容不能为空")

        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        prepared_chunks: list[PreparedChunk] = []

        for heading_path, section in self._sections(normalized):
            for chunk_content in self._splitter.split_text(section):
                prepared_chunks.append(
                    PreparedChunk(
                        content=chunk_content,
                        document_name=filename,
                        heading_path=heading_path,
                        chunk_index=len(prepared_chunks),
                    )
                )

        return PreparedDocument(
            filename=filename,
            normalized_content=normalized,
            content_hash=content_hash,
            chunks=tuple(prepared_chunks),
        )

    def _sections(self, content: str) -> list[tuple[tuple[str, ...], str]]:
        sections: list[tuple[tuple[str, ...], str]] = []
        heading_path: list[str] = []
        body: list[str] = []

        def append_section() -> None:
            text = "\n".join(body).strip()
            if text:
                sections.append((tuple(heading_path), text))

        for line in content.split("\n"):
            match = self._heading_pattern.match(line)
            if not match:
                body.append(line)
                continue

            append_section()
            body.clear()
            level = len(match.group(1))
            heading_path[level - 1 :] = [match.group(2).strip()]

        append_section()
        return sections
