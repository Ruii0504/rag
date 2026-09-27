import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """应用配置无效。"""


@dataclass(frozen=True)
class Settings:
    qwen_api_key: str
    qwen_base_url: str
    qwen_chat_model: str
    qwen_embedding_model: str
    qwen_embedding_dimension: int
    sqlite_path: Path
    chroma_path: Path
    uploads_path: Path
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    retrieval_score_threshold: float
    qwen_rerank_model: str = "qwen3-rerank"
    qwen_rerank_url: str = "https://maas.qianwenaiapi.com/compatible-api/v1/reranks"

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "Settings":
        root = Path(base_dir or Path.cwd()).resolve()
        load_dotenv(root / ".env", override=False)

        api_key = os.getenv("QWEN_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError("缺少必需的环境变量 QWEN_API_KEY")

        def read_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} 必须是整数") from exc

        def read_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} 必须是数字") from exc

        def read_path(name: str, default: str) -> Path:
            value = Path(os.getenv(name, default))
            return value if value.is_absolute() else root / value

        dimension = read_int("QWEN_EMBEDDING_DIMENSION", 1024)
        chunk_size = read_int("CHUNK_SIZE", 800)
        chunk_overlap = read_int("CHUNK_OVERLAP", 100)
        top_k = read_int("RETRIEVAL_TOP_K", 7)
        threshold = read_float("RETRIEVAL_SCORE_THRESHOLD", 0.5)

        if dimension <= 0:
            raise ConfigurationError("QWEN_EMBEDDING_DIMENSION 必须大于 0")
        if chunk_size <= 0:
            raise ConfigurationError("CHUNK_SIZE 必须大于 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ConfigurationError("CHUNK_OVERLAP 必须大于等于 0 且小于 CHUNK_SIZE")
        if top_k <= 0:
            raise ConfigurationError("RETRIEVAL_TOP_K 必须大于 0")
        if not 0 <= threshold <= 1:
            raise ConfigurationError("RETRIEVAL_SCORE_THRESHOLD 必须在 0 到 1 之间")

        return cls(
            qwen_api_key=api_key,
            qwen_base_url=os.getenv(
                "QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            qwen_chat_model=os.getenv("QWEN_CHAT_MODEL", "qwen-plus"),
            qwen_embedding_model=os.getenv(
                "QWEN_EMBEDDING_MODEL",
                "text-embedding-v4",
            ),
            qwen_embedding_dimension=dimension,
            sqlite_path=read_path("SQLITE_PATH", "data/app.db"),
            chroma_path=read_path("CHROMA_PATH", "data/chroma"),
            uploads_path=read_path("UPLOADS_PATH", "data/uploads"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_top_k=top_k,
            retrieval_score_threshold=threshold,
            qwen_rerank_url=os.getenv(
                "QWEN_RERANK_URL",
                "https://maas.qianwenaiapi.com/compatible-api/v1/reranks",
            ),
        )
