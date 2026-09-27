import { FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react"

import {
  createKnowledgeBase,
  deleteDocument,
  deleteKnowledgeBase,
  listDocuments,
  listKnowledgeBases,
  loadStatus,
  streamChat,
  uploadDocuments,
} from "./api"
import type {
  ChatHistoryMessage,
  DocumentSummary,
  EvidenceSummary,
  KnowledgeBaseSummary,
  SystemStatus,
  TraceEvent,
} from "./types"


const Icon = ({ name, size = 18 }: { name: string; size?: number }) => {
  const paths: Record<string, ReactNode> = {
    home: <><path d="m3 10 9-7 9 7v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 21v-6h6v6"/></>,
    database: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/></>,
    message: <path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 9.4 9.4 0 0 1-4-.9L3 21l1.6-4.2A8.5 8.5 0 1 1 21 11.5Z"/>,
    plus: <path d="M12 5v14M5 12h14"/>,
    chevron: <path d="m7 10 5 5 5-5"/>,
    upload: <><path d="M12 16V3"/><path d="m7 8 5-5 5 5"/><path d="M5 21h14"/></>,
    send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>,
    more: <><circle cx="5" cy="12" r="1" fill="currentColor"/><circle cx="12" cy="12" r="1" fill="currentColor"/><circle cx="19" cy="12" r="1" fill="currentColor"/></>,
    close: <path d="m6 6 12 12M18 6 6 18"/>,
    check: <path d="m5 12 4 4L19 6"/>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h5"/></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

const displayDate = (value: string) => new Intl.DateTimeFormat("zh-CN", {
  dateStyle: "short",
  timeStyle: "short",
  hour12: false,
}).format(new Date(value))

export default function App() {
  const [page, setPage] = useState<"chat" | "knowledge" | "home">("home")
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [bases, setBases] = useState<KnowledgeBaseSummary[]>([])
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [activeBaseId, setActiveBaseId] = useState<string | null>(null)
  const [chatBaseIds, setChatBaseIds] = useState<string[]>([])
  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState("")
  const [chatState, setChatState] = useState<"idle" | "thinking" | "done" | "refused" | "error">("idle")
  const [history, setHistory] = useState<ChatHistoryMessage[]>([])
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([])
  const [evidences, setEvidences] = useState<EvidenceSummary[]>([])
  const [newBase, setNewBase] = useState("")
  const [modal, setModal] = useState<"create" | "delete" | "deleteDocument" | "upload" | null>(null)
  const [documentToDelete, setDocumentToDelete] = useState<DocumentSummary | null>(null)
  const [uploadFiles, setUploadFiles] = useState<File[]>([])
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState("")
  const [error, setError] = useState("")
  const chatController = useRef<AbortController | null>(null)
  const chatGeneration = useRef(0)

  const activeBase = useMemo(
    () => bases.find(base => base.id === activeBaseId),
    [activeBaseId, bases],
  )
  const supportingEvidenceCount = useMemo(
    () => evidences.filter(evidence => evidence.support_status === "supporting").length,
    [evidences],
  )
  const activeStage = traceEvents[traceEvents.length - 1]?.stage
  const thinkingMessage = activeStage === "validate"
    ? "正在核验答案与证据…"
    : activeStage === "generate"
      ? "正在生成候选答案…"
      : activeStage === "judge"
        ? "正在判定资料能否回答…"
        : "正在检索已选知识库…"

  useEffect(() => {
    Promise.all([loadStatus(), listKnowledgeBases()])
      .then(([nextStatus, nextBases]) => {
        setStatus(nextStatus)
        setBases(nextBases)
        setActiveBaseId(current => current ?? nextBases[0]?.id ?? null)
        setChatBaseIds(current => current.length ? current : nextBases[0] ? [nextBases[0].id] : [])
      })
      .catch(apiError => setError(apiError instanceof Error ? apiError.message : "应用加载失败"))
  }, [])

  useEffect(() => {
    const kbId = activeBaseId
    if (!kbId) {
      setDocuments([])
      return
    }
    let current = true
    listDocuments(kbId)
      .then(nextDocuments => {
        if (current) setDocuments(nextDocuments)
      })
      .catch(apiError => {
        if (current) setError(apiError instanceof Error ? apiError.message : "资料加载失败")
      })
    return () => {
      current = false
    }
  }, [activeBaseId])

  const notify = (text: string) => {
    setToast(text)
    window.setTimeout(() => setToast(""), 2500)
  }

  const toggleBase = (id: string) => {
    setChatBaseIds(current => current.includes(id)
      ? current.filter(currentId => currentId !== id)
      : [...current, id])
  }

  const submitCreateBase = async () => {
    const name = newBase.trim()
    if (!name) return
    try {
      const created = await createKnowledgeBase(name)
      setBases(current => [...current, created])
      setActiveBaseId(created.id)
      setChatBaseIds(current => current.length ? current : [created.id])
      setNewBase("")
      setModal(null)
      notify("知识库已创建")
    } catch (apiError) {
      notify(apiError instanceof Error ? apiError.message : "知识库创建失败")
    }
  }

  const submitUpload = async () => {
    if (!activeBase || !uploadFiles.length) return
    setBusy(true)
    try {
      const result = await uploadDocuments(activeBase.id, uploadFiles)
      const uploaded = result.documents
      setDocuments(current => [...current, ...uploaded])
      const addedChunks = uploaded.reduce((sum, document) => sum + document.chunk_count, 0)
      setBases(current => current.map(base => base.id === activeBase.id ? {
        ...base,
        document_count: base.document_count + uploaded.length,
        chunk_count: base.chunk_count + addedChunks,
      } : base))
      setStatus(current => current ? {
        ...current,
        document_count: current.document_count + uploaded.length,
        chunk_count: current.chunk_count + addedChunks,
      } : current)
      setUploadFiles([])
      setModal(null)
      notify(result.errors.length
        ? `${uploaded.length} 份成功，${result.errors.length} 份失败`
        : "文档索引已完成")
    } catch (apiError) {
      notify(apiError instanceof Error ? apiError.message : "文档上传失败")
    } finally {
      setBusy(false)
    }
  }

  const submitDeleteBase = async () => {
    if (!activeBase) return
    setBusy(true)
    try {
      await deleteKnowledgeBase(activeBase.id)
      const removed = activeBase
      const remaining = bases.filter(base => base.id !== removed.id)
      setBases(remaining)
      setActiveBaseId(remaining[0]?.id ?? null)
      setChatBaseIds(current => {
        const retained = current.filter(id => id !== removed.id)
        return retained.length ? retained : remaining[0] ? [remaining[0].id] : []
      })
      setStatus(current => current ? {
        ...current,
        knowledge_base_count: Math.max(0, current.knowledge_base_count - 1),
        document_count: Math.max(0, current.document_count - removed.document_count),
        chunk_count: Math.max(0, current.chunk_count - removed.chunk_count),
      } : current)
      setModal(null)
      notify("知识库及其资料已移除")
    } catch (apiError) {
      notify(apiError instanceof Error ? apiError.message : "知识库删除失败")
    } finally {
      setBusy(false)
    }
  }

  const submitDeleteDocument = async () => {
    if (!activeBase || !documentToDelete) return
    setBusy(true)
    try {
      await deleteDocument(activeBase.id, documentToDelete.id)
      setDocuments(current => current.filter(document => document.id !== documentToDelete.id))
      setBases(current => current.map(base => base.id === activeBase.id ? {
        ...base,
        document_count: Math.max(0, base.document_count - 1),
        chunk_count: Math.max(0, base.chunk_count - documentToDelete.chunk_count),
      } : base))
      setStatus(current => current ? {
        ...current,
        document_count: Math.max(0, current.document_count - 1),
        chunk_count: Math.max(0, current.chunk_count - documentToDelete.chunk_count),
      } : current)
      setDocumentToDelete(null)
      setModal(null)
      notify("文档已删除")
    } catch (apiError) {
      notify(apiError instanceof Error ? apiError.message : "文档删除失败")
    } finally {
      setBusy(false)
    }
  }

  const ask = async (event: FormEvent) => {
    event.preventDefault()
    const submittedQuestion = question.trim()
    if (!chatBaseIds.length) {
      notify("请先选择至少一个知识库")
      return
    }
    if (!submittedQuestion || chatState === "thinking") return

    const generation = chatGeneration.current + 1
    chatGeneration.current = generation
    const controller = new AbortController()
    chatController.current = controller
    const priorHistory = history
    setQuestion("")
    setAnswer("")
    setHistory(current => [...current, { role: "user", content: submittedQuestion }])
    setTraceEvents([])
    setEvidences([])
    setChatState("thinking")
    let answerText = ""
    let refused = false
    let failed = false

    try {
      await streamChat(
        { question: submittedQuestion, history: priorHistory, kb_ids: chatBaseIds },
        traceEvent => {
          if (generation !== chatGeneration.current) return
          if (traceEvent.type !== "answer_delta" && traceEvent.type !== "completed") {
            setTraceEvents(current => [...current, traceEvent])
          }
          if (traceEvent.type === "evidence") {
            setEvidences(traceEvent.payload.result?.evidences ?? [])
          }
          if (traceEvent.type === "answer_delta") {
            refused ||= traceEvent.stage === "refuse"
            answerText += traceEvent.message
            setAnswer(answerText)
          }
          if (traceEvent.type === "error") {
            failed = true
            answerText = traceEvent.message
            setAnswer(traceEvent.message)
            setChatState("error")
          }
        },
        controller.signal,
      )
      if (generation !== chatGeneration.current) return
      if (!failed) setChatState(refused ? "refused" : "done")
      if (answerText) {
        setHistory(current => [
          ...current,
          { role: "assistant", content: answerText },
        ])
      }
    } catch (apiError) {
      if (
        generation !== chatGeneration.current
        || (apiError instanceof Error && apiError.name === "AbortError")
      ) return
      const message = apiError instanceof Error ? apiError.message : "客服请求失败"
      setAnswer(message)
      setHistory(current => [...current, { role: "assistant", content: message }])
      setChatState("error")
    } finally {
      if (chatController.current === controller) chatController.current = null
    }
  }

  const startNewChat = () => {
    chatGeneration.current += 1
    chatController.current?.abort()
    chatController.current = null
    setQuestion("")
    setAnswer("")
    setHistory([])
    setTraceEvents([])
    setEvidences([])
    setChatState("idle")
  }

  return <main className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={() => setPage("home")}><span className="brand-mark"><i/><i/><i/></span><span>evidence<span className="brand-dim">.local</span></span></button>
      <div className="system-state"><span className="status-dot"/>{status ? "系统就绪" : "系统连接中"} <span className="thin-divider"/> {status?.chat_model ?? "—"} <span className="thin-divider"/> {status?.embedding_model ?? "—"}</div>
      <button className="profile">内部工作台 <span className="avatar">L</span><Icon name="chevron" size={14}/></button>
    </header>

    <div className="workspace">
      <aside className="rail">
        <nav>
          <button className={page === "home" ? "nav-item active" : "nav-item"} onClick={() => setPage("home")}><Icon name="home"/>概览</button>
          <button className={page === "knowledge" ? "nav-item active" : "nav-item"} onClick={() => setPage("knowledge")}><Icon name="database"/>知识库</button>
          <button className={page === "chat" ? "nav-item active" : "nav-item"} onClick={() => setPage("chat")}><Icon name="message"/>智能客服</button>
        </nav>
        <div className="rail-footer"><span className="eyebrow">STORAGE</span><span>{status?.chunk_count ?? 0} chunks indexed</span><div className="usage"><i/></div><small>{status?.document_count ?? 0} documents / local</small></div>
      </aside>

      {page === "home" && <section className="home-page">
        <div className="home-gridline"/>
        <div className="home-hero"><span className="eyebrow">LOCAL RAG / CUSTOMER SUPPORT</span><h1>每一次回答，<em>都有据可循。</em></h1><p>把分散的产品资料变成可验证的客服知识。系统在回答前强制检索，并将每个结论链接回原始文档。</p><div className="hero-actions"><button className="lime-button" onClick={() => setPage("chat")}>开始问答 <Icon name="send" size={15}/></button><button className="outline-button" onClick={() => setPage("knowledge")}>管理知识库</button></div></div>
        <div className="model-console"><div className="console-head"><span><b/><b/><b/></span><small>runtime / config</small></div><div className="console-content"><p><span>01</span><strong>chat_model</strong> <i>=</i> <u>"{status?.chat_model ?? "loading"}"</u><mark>{status ? "ready" : "pending"}</mark></p><p><span>02</span><strong>embedding</strong> <i>=</i> <u>"{status?.embedding_model ?? "loading"}"</u><mark>{status ? "ready" : "pending"}</mark></p><p><span>03</span><strong>api_key</strong> <i>=</i> <u>"••••••••••••••••"</u><mark>secured</mark></p></div></div>
      </section>}

      {page === "knowledge" && <section className="knowledge-page">
        <div className="page-heading"><div><span className="eyebrow">KNOWLEDGE MANAGEMENT</span><h1>知识库与资料</h1><p>上传 Markdown 产品资料，按标题切分并建立可追溯的检索索引。</p></div><button className="lime-button" onClick={() => setModal("create")}><Icon name="plus" size={17}/>新建知识库</button></div>
        <div className="knowledge-layout">
          <div className="bases-card"><div className="card-title">知识库 <span>{bases.length}</span></div>{bases.map(base => <button key={base.id} onClick={() => setActiveBaseId(base.id)} className={activeBaseId === base.id ? "base-row selected" : "base-row"}><span className="database-icon"><Icon name="database" size={16}/></span><span><b>{base.name}</b><small>{base.document_count} 份文档 · {base.chunk_count} chunks</small></span><Icon name="more" size={17}/></button>)}</div>
          <div className="docs-panel"><div className="docs-top"><div><span className="eyebrow">ACTIVE COLLECTION</span><h2>{activeBase?.name ?? "尚未选择知识库"}</h2></div><div className="docs-actions"><button className="outline-button compact" disabled={!activeBase} onClick={() => setModal("delete")}>删除</button><button className="outline-button compact" disabled={!activeBase} onClick={() => setModal("upload")}><Icon name="upload" size={16}/>上传 Markdown</button></div></div>{activeBase ? <><div className="dropzone" onClick={() => setModal("upload")}><Icon name="upload" size={22}/><b>拖放 Markdown 文件至此处</b><span>或点击浏览 · 仅支持 UTF-8 .md</span></div><div className="doc-table"><div className="table-head"><span>文件名</span><span>状态</span><span>文本块</span><span>上传时间</span></div>{documents.map(document => <div className="table-row" key={document.id}><span><Icon name="file" size={16}/>{document.filename}<button className="document-delete" aria-label={`删除 ${document.filename}`} onClick={() => { setDocumentToDelete(document); setModal("deleteDocument") }}><Icon name="close" size={14}/></button></span><span className="completed"><i/>{document.status === "ready" ? "已完成" : document.status}</span><span>{document.chunk_count}</span><span>{displayDate(document.created_at)}</span></div>)}</div></> : <div className="empty-state"><Icon name="database" size={30}/><h3>选择一个知识库</h3><p>选择后即可查看资料与上传新文档。</p></div>}</div>
        </div>
      </section>}

      {page === "chat" && <section className="chat-page">
        <div className="chat-header"><div><span className="eyebrow">EVIDENCE-BOUND ANSWERING</span><h1>智能客服</h1></div><button className="outline-button compact" onClick={startNewChat}>新建会话 <Icon name="plus" size={15}/></button></div>
        <div className="chat-layout">
          <aside className="collection-panel"><div className="panel-label"><span>检索范围</span><small>至少选择 1 个</small></div>{bases.map(base => <label className="collection-choice" key={base.id}><input type="checkbox" checked={chatBaseIds.includes(base.id)} onChange={() => toggleBase(base.id)}/><span className="fake-check"><Icon name="check" size={12}/></span><span><b>{base.name}</b><small>{base.document_count} docs · {base.chunk_count} chunks</small></span></label>)}<div className="threshold"><span>{status?.rerank_model ? "重排保留" : "候选阈值"}</span><b>{status?.rerank_model ? `Top ${status.retrieval_top_k}` : status?.retrieval_score_threshold.toFixed(2) ?? "—"}</b><div><i/></div><small>{status?.rerank_model ? `${status.rerank_model} · 能否回答仍由语义判定确认` : "阈值用于筛选候选，能否回答由语义判定确认"}</small></div></aside>
          <div className="conversation">
            <div className="messages">
              <div className="welcome"><span className="signal-icon"><i/><i/><i/></span><h2>需要查找什么资料？</h2><p>我会先在选定知识库中检索，再仅基于可靠证据回答。</p><div className="prompt-chips"><button onClick={() => setQuestion("如何恢复路由器出厂设置？")}>如何恢复出厂设置？</button><button onClick={() => setQuestion("Mesh 组网最多支持几台设备？")}>Mesh 组网数量限制</button></div></div>
              {history.map((message, index) => message.role === "user"
                ? <div className="user-message" key={`message-${index}`}>{message.content}</div>
                : <div className="assistant-message" key={`message-${index}`}>
                    <div className="assistant-id"><span className="mini-mark">E</span> Evidence Assistant <small>刚刚</small></div>
                    <p>{message.content}</p>
                    {index === history.length - 1 && <div className="answer-footer"><span>{chatState === "refused" ? "证据不足 · 已拒答" : chatState === "error" ? "处理失败" : `回答完成 · ${supportingEvidenceCount} 条引用`}</span><button type="button" onClick={() => navigator.clipboard?.writeText(message.content)}>复制回答</button></div>}
                  </div>)}
              {chatState === "thinking" && <div className="assistant-message">
                <div className="assistant-id"><span className="mini-mark">E</span> Evidence Assistant <small>刚刚</small></div>
                {!answer ? <div className="typing"><i/><i/><i/> {thinkingMessage}</div> : <p>{answer}</p>}
              </div>}
            </div>
            <form className="composer" onSubmit={ask}><textarea aria-label="客户问题" value={question} onChange={event => setQuestion(event.target.value)} placeholder={chatBaseIds.length ? "输入客户问题…" : "请先在左侧选择知识库"} disabled={!chatBaseIds.length || chatState === "thinking"}/><div><span><kbd>⌘</kbd><kbd>↵</kbd> 发送</span><button aria-label="发送问题" disabled={!chatBaseIds.length || chatState === "thinking"}><Icon name="send" size={18}/></button></div></form>
          </div>
          <aside className="evidence-panel">
            <div className="evidence-title"><span>执行过程</span><small>本次会话</small></div>
            {traceEvents.length || evidences.length ? <>
              <div className="timeline">{traceEvents.map((traceEvent, index) => <div className="event done" key={`${traceEvent.type}-${traceEvent.stage}-${index}`}><i/><span>{traceEvent.message}</span><small>{String(index + 1).padStart(2, "0")}</small></div>)}</div>
              <div className="evidence-title source-head"><span>检索资料</span><small>{evidences.length}</small></div>
              {evidences.map(evidence => <article className={`evidence-card ${evidence.support_status}`} key={`${evidence.reference_id}-${evidence.document_id}-${evidence.chunk_index}`}><div className="evidence-meta"><span className="evidence-num">{evidence.reference_id}</span><b>{evidence.rerank_score != null ? `重排 ${evidence.rerank_score.toFixed(2)}` : evidence.similarity.toFixed(2)}</b><span className={`evidence-status ${evidence.support_status}`}>{evidence.support_status === "supporting" ? "回答证据" : "相关资料，不足以直接回答"}</span></div>{evidence.rerank_score != null && <small>向量 {evidence.similarity.toFixed(2)}</small>}<p className="evidence-question">对应：{evidence.subquestion}</p><h3>{evidence.document_name}</h3><p className="path">{evidence.heading_path.join(" / ") || "无标题"}</p><blockquote>{evidence.content}</blockquote></article>)}
            </> : <div className="empty-state"><Icon name="message" size={28}/><h3>等待提问</h3><p>回答过程与引用证据会显示在这里。</p></div>}
          </aside>
        </div>
      </section>}
    </div>

    {modal && <div className="modal-backdrop" onMouseDown={() => setModal(null)}><div className="modal" onMouseDown={event => event.stopPropagation()}><button aria-label="关闭" className="modal-close" onClick={() => setModal(null)}><Icon name="close"/></button>{modal === "create" ? <><span className="eyebrow">NEW COLLECTION</span><h2>创建知识库</h2><p>知识库用于限制检索范围，名称创建后仍可在此处管理。</p><label className="field-label">知识库名称<input autoFocus value={newBase} onChange={event => setNewBase(event.target.value)} placeholder="例如：智能家居产品资料"/></label><button className="lime-button full" onClick={submitCreateBase}>创建知识库</button></> : modal === "upload" ? <><span className="eyebrow">ADD DOCUMENTS</span><h2>上传 Markdown</h2><label className="modal-upload"><Icon name="upload" size={28}/><b>{uploadFiles.length ? `已选择 ${uploadFiles.length} 份文件` : "选择 .md 文件"}</b><small>支持一次上传多份 UTF-8 Markdown 文档</small><input aria-label="Markdown 文件" type="file" accept=".md,text/markdown" multiple onChange={event => setUploadFiles(Array.from(event.target.files ?? []))}/></label><button className="lime-button full" disabled={!uploadFiles.length || busy} onClick={submitUpload}>{busy ? "正在建立索引…" : "开始上传"}</button></> : modal === "deleteDocument" ? <><span className="eyebrow">CONFIRM REMOVAL</span><h2>删除文档？</h2><p>将永久删除 <b>{documentToDelete?.filename}</b> 的原文件、元数据及向量块。此操作无法撤销。</p><div className="modal-actions"><button className="outline-button" onClick={() => setModal(null)}>取消</button><button className="danger-button" disabled={busy} onClick={submitDeleteDocument}>{busy ? "正在删除…" : "确认删除文档"}</button></div></> : <><span className="eyebrow">CONFIRM REMOVAL</span><h2>删除知识库？</h2><p>将永久删除 <b>{activeBase?.name}</b> 中的文档、元数据及全部向量块。此操作无法撤销。</p><div className="modal-actions"><button className="outline-button" onClick={() => setModal(null)}>取消</button><button className="danger-button" disabled={busy} onClick={submitDeleteBase}>{busy ? "正在删除…" : "确认删除"}</button></div></>}</div></div>}
    {error && <div className="toast"><span className="status-dot"/>{error}</div>}
    {toast && <div className="toast"><span className="status-dot"/>{toast}</div>}
  </main>
}
