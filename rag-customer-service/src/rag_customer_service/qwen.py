from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any
import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_customer_service.config import Settings
from rag_customer_service.model_diagnostics import record_call


class QwenClientError(RuntimeError):
    """可安全展示的千问调用错误。"""


class QwenClient:
    _embedding_batch_size = 10

    def __init__(
        self,
        settings: Settings,
        chat_factory: Callable[..., Any] = ChatOpenAI,
        embeddings_factory: Callable[..., Any] = OpenAIEmbeddings,
    ) -> None:
        self._settings = settings
        self.chat_model = chat_factory(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            model=settings.qwen_chat_model,
            temperature=0,
            timeout=30.0,
            max_retries=1,
        )
        self.embeddings = embeddings_factory(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            model=settings.qwen_embedding_model,
            dimensions=settings.qwen_embedding_dimension,
            check_embedding_ctx_length=False,
            request_timeout=30.0,
            max_retries=1,
        )

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        started = time.perf_counter()
        try:
            response = self.chat_model.invoke(messages)
            record_call("chat", self._settings.qwen_chat_model, started,
                        usage=getattr(response, "usage_metadata", None))
            return str(response.content)
        except Exception as error:
            record_call("chat", self._settings.qwen_chat_model, started, error=error, secret=self._settings.qwen_api_key)
            raise QwenClientError(f"千问聊天调用失败（{type(error).__name__}），请检查模型配置和网络连接") from None

    def stream_chat(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> Iterable[str]:
        try:
            for chunk in self.chat_model.stream(messages):
                content = str(chunk.content)
                if content:
                    yield content
        except Exception:
            raise QwenClientError("千问流式调用失败，请检查模型配置和网络连接") from None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        try:
            embeddings: list[list[float]] = []
            for start in range(0, len(texts), self._embedding_batch_size):
                embeddings.extend(
                    self.embeddings.embed_documents(
                        texts[start : start + self._embedding_batch_size]
                    )
                )
            record_call("embed_documents", self._settings.qwen_embedding_model, started)
            return embeddings
        except Exception as error:
            record_call("embed_documents", self._settings.qwen_embedding_model, started, error=error, secret=self._settings.qwen_api_key)
            raise QwenClientError("千问文档向量化失败，请检查模型配置和网络连接") from None

    def embed_query(self, text: str) -> list[float]:
        started = time.perf_counter()
        try:
            result = self.embeddings.embed_query(text)
            record_call("embed_query", self._settings.qwen_embedding_model, started)
            return result
        except Exception as error:
            record_call("embed_query", self._settings.qwen_embedding_model, started, error=error, secret=self._settings.qwen_api_key)
            raise QwenClientError("千问查询向量化失败，请检查模型配置和网络连接") from None
