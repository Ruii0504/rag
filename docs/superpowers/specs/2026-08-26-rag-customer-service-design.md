# RAG 智能客服系统设计基线

## 目标

构建一个供内部单实例使用的学习型 RAG 智能客服系统。系统使用 Streamlit 提供知识库管理和智能客服页面，使用 LangGraph 编排确定性 RAG 工作流，使用 Chroma 保存向量，使用 SQLite 保存知识库及文档元数据，并通过千问官方 OpenAI 兼容接口调用聊天和 Embedding 模型。

## 已确认范围

- 单机、单实例、无登录、少量并发。
- 总数据量不超过数万文本块。
- 只接收 Markdown 文档。
- 支持知识库列表、新建和删除。
- 支持文档列表、上传和删除。
- 同一知识库内按内容哈希阻止重复文档，不同知识库允许相同内容。
- 所有向量保存在一个 Chroma collection，通过 `kb_id` 过滤多个已选知识库。
- Markdown 使用标题感知切分，超长章节继续递归切分。
- 上传后同步执行切分、向量化和入库，并向界面报告进度。
- 对话历史只保存在当前 Streamlit 会话中。
- 每次提问都执行检索，不由模型决定是否检索。
- Agent 流程固定为“问题改写 → 检索 → 证据判定 → 回答或拒答”。
- 证据判定使用 Top-K 与可配置相似度阈值，不额外调用模型判定。
- 回答必须基于检索证据，并展示文档名、章节、相似度和文本片段。
- 没有充分证据时明确拒答，不使用模型自身知识补充。

## 明确不做

- 用户登录、角色权限和多租户隔离。
- FastAPI、Celery、Redis 和后台任务队列。
- 对话历史持久化。
- 混合检索、重排序器和知识图谱。
- 多模型动态切换。
- 分布式 Chroma 或独立向量数据库服务。
- PDF、Word、网页等非 Markdown 数据源。

## 推荐结构

```text
rag-customer-service/
├── app.py
├── pages/
│   ├── 1_知识库管理.py
│   └── 2_智能客服对话.py
├── src/rag_customer_service/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── qwen.py
│   ├── storage.py
│   ├── vector_store.py
│   ├── ingestion.py
│   ├── knowledge_base.py
│   ├── retrieval.py
│   ├── agent.py
│   └── bootstrap.py
├── data/
│   ├── app.db
│   ├── chroma/
│   └── uploads/
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

`pyproject.toml` 只负责让 `src/` 包可通过 editable install 正确发现；依赖版本继续由 `requirements.txt` 管理。

运行时基线使用 Python 3.12（经用户于 2026-08-26 确认）。

## 核心数据流

### 文档摄取

页面提交文件后，知识库模块依次执行：格式及内容校验、文本规范化、内容哈希、同库查重、标题感知切分、批量 Embedding、保存原文件、写入 Chroma、写入 SQLite。任一步失败时，删除本次已经产生的文件、向量和元数据。

### 客服问答

页面提交问题和已选知识库 ID。LangGraph 使用会话历史把问题改写为独立问题，随后强制调用检索工具。检索模块生成查询向量，在单一 collection 中按 `kb_id` 过滤并执行 Top-K 与阈值判断。无充分证据时进入拒答节点；有充分证据时把编号证据交给千问流式生成答案。

## 持久化约束

- SQLite 至少包含 `knowledge_bases` 和 `documents` 两张表。
- `documents` 对 `(kb_id, content_hash)` 建立唯一约束。
- Chroma 块元数据至少包含 `kb_id`、`document_id`、`document_name`、`heading_path`、`chunk_index` 和 `content_hash`。
- 同一 collection 中所有向量必须来自相同 Embedding 模型和相同维度。
- 更换 Embedding 模型或维度必须清空并重建全部向量。
- 删除文档后，其原文件、SQLite 元数据和 Chroma 文本块均不可残留。

## 验收标准

- 能通过界面创建和删除知识库。
- 能上传现有产品 Markdown 文档并看到处理进度。
- 重复上传同一内容时被明确拦截。
- 多选知识库后，检索结果只来自所选知识库。
- 回答包含来源文档与章节引用。
- 无证据问题明确拒答。
- 删除文档或知识库后，相关内容不能再次被召回。
- 页面刷新前的多轮追问能够利用当前会话历史完成问题改写。
