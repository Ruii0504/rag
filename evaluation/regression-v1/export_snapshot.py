"""Export the existing corpus using read-only SQLite connections; no model calls."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def connect(path):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def export():
    with connect(ROOT / "rag-customer-service/data/app.db") as db:
        documents = [dict(row) for row in db.execute("SELECT * FROM documents ORDER BY filename")]
    assert len(documents) == 3
    for doc in documents:
        source = ROOT / "product data" / doc["filename"]
        original = source.read_text(encoding="utf-8")
        uploaded = Path(doc["file_path"]).read_text(encoding="utf-8")
        assert digest(original) == doc["content_hash"] == digest(uploaded), doc["filename"]
        doc["source_path"] = source.relative_to(ROOT).as_posix()
    by_id = {doc["id"]: doc for doc in documents}
    with connect(ROOT / "rag-customer-service/data/chroma/chroma.sqlite3") as db:
        rows = db.execute("""SELECT e.embedding_id, m.key, m.string_value, m.int_value
            FROM embeddings e JOIN embedding_metadata m ON m.id=e.id
            JOIN segments s ON s.id=e.segment_id JOIN collections c ON c.id=s.collection
            WHERE c.name='rag_chunks'""").fetchall()
        entries = {}
        for row in rows:
            entries.setdefault(row["embedding_id"], {})[row["key"]] = (
                row["string_value"] if row["string_value"] is not None else row["int_value"]
            )
        chunks = []
        for chunk_id, metadata in entries.items():
            if metadata["document_id"] not in by_id:
                continue
            content = metadata["chroma:document"]
            chunks.append({"chunk_id": chunk_id, "document_id": metadata["document_id"],
                "document_name": metadata["document_name"], "kb_id": metadata["kb_id"],
                "chunk_index": metadata["chunk_index"], "heading_path": json.loads(metadata["heading_path"]),
                "document_content_hash": metadata["content_hash"], "chunk_sha256": digest(content),
                "content": content})
    chunks.sort(key=lambda row: (row["document_name"], row["chunk_index"]))
    assert len(chunks) == 55 == sum(doc["chunk_count"] for doc in documents)
    for chunk in chunks:
        assert chunk["chunk_id"] == f'{chunk["document_id"]}:{chunk["chunk_index"]}'
        assert chunk["document_content_hash"] == by_id[chunk["document_id"]]["content_hash"]
    payload = "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
    (HERE / "chunks.snapshot.jsonl").write_text(payload, encoding="utf-8", newline="\n")
    manifest = {"dataset_version": "1.0", "snapshot_at": datetime.now(timezone.utc).isoformat(),
        "source": "Read-only SQLite metadata and Chroma persisted document text",
        "chunk_count": len(chunks), "documents": documents, "snapshot_sha256": digest(payload),
        "embedding_model": "text-embedding-v4", "embedding_dimension": 1024,
        "annotation_status": "agent_authored_pending_user_review", "model_evaluation_run": False,
        "frequency_basis": "User-assigned 60/30/10 sampling quotas; not measured production traffic"}
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 实际入库片段快照", "", "只读导出；编号为文档序号:零起始 chunk_index，真实 ID 见各节。", ""]
    for chunk in chunks:
        lines.extend([f'## {chunk["document_name"][:2]}:{chunk["chunk_index"]}',
            f'- chunk_id: `{chunk["chunk_id"]}`', f'- 章节：{" / ".join(chunk["heading_path"])}',
            "", chunk["content"], ""])
    (HERE / "chunks.snapshot.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"documents": [{"name": d["filename"], "id": d["id"], "chunks": d["chunk_count"]} for d in documents],
        "chunks": len(chunks), "source_hashes_match": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    export()
