import { FormEvent, type ReactNode, useState } from "react"

type KnowledgeBase = { id: string; name: string; docs: number; chunks: number }

const initialBases: KnowledgeBase[] = [
  { id: "router", name: "路由器产品知识", docs: 6, chunks: 128 },
  { id: "after-sales", name: "售后与保修政策", docs: 3, chunks: 42 },
  { id: "network", name: "网络部署手册", docs: 4, chunks: 76 },
]

const Icon = ({ name, size = 18 }: { name: string; size?: number }) => {
  const paths: Record<string, ReactNode> = {
    home: <><path d="m3 10 9-7 9 7v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 21v-6h6v6"/></>,
    database: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/></>,
    message: <><path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 9.4 9.4 0 0 1-4-.9L3 21l1.6-4.2A8.5 8.5 0 1 1 21 11.5Z"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    chevron: <path d="m7 10 5 5 5-5"/>,
    upload: <><path d="M12 16V3"/><path d="m7 8 5-5 5 5"/><path d="M5 21h14"/></>,
    send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>,
    search: <><circle cx="11" cy="11" r="6"/><path d="m20 20-4-4"/></>,
    more: <><circle cx="5" cy="12" r="1" fill="currentColor"/><circle cx="12" cy="12" r="1" fill="currentColor"/><circle cx="19" cy="12" r="1" fill="currentColor"/></>,
    close: <><path d="m6 6 12 12M18 6 6 18"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h5"/></>,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

export default function App() {
  const [page, setPage] = useState<"chat" | "knowledge" | "home">("chat")
  const [bases, setBases] = useState(initialBases)
  const [selected, setSelected] = useState<string[]>(["router"])
  const [question, setQuestion] = useState("")
  const [answerState, setAnswerState] = useState<"idle" | "thinking" | "done" | "refused">("done")
  const [newBase, setNewBase] = useState("")
  const [modal, setModal] = useState<"create" | "delete" | "upload" | null>(null)
  const [toast, setToast] = useState("")

  const toggleBase = (id: string) => setSelected(current => current.includes(id) ? current.filter(x => x !== id) : [...current, id])
  const notify = (text: string) => { setToast(text); window.setTimeout(() => setToast(""), 2500) }
  const ask = (e: FormEvent) => {
    e.preventDefault()
    if (!selected.length) return notify("请先选择至少一个知识库")
    if (!question.trim()) return
    setAnswerState("thinking")
    window.setTimeout(() => setAnswerState(question.includes("天气") ? "refused" : "done"), 900)
  }
  const createBase = () => {
    const name = newBase.trim()
    if (!name) return
    if (bases.some(b => b.name === name)) return notify("知识库名称已存在")
    setBases([...bases, { id: String(Date.now()), name, docs: 0, chunks: 0 }])
    setNewBase(""); setModal(null); notify("知识库已创建")
  }
  const deleteBase = () => {
    const target = selected[0]
    setBases(bases.filter(b => b.id !== target)); setSelected([]); setModal(null); notify("知识库及其资料已移除")
  }
  const activeBase = bases.find(b => b.id === selected[0])

  return <main className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={() => setPage("home")}><span className="brand-mark"><i/><i/><i/></span><span>evidence<span className="brand-dim">.local</span></span></button>
      <div className="system-state"><span className="status-dot"/>系统就绪 <span className="thin-divider"/> Qwen-Plus <span className="thin-divider"/> text-embedding-v3</div>
      <button className="profile">内部工作台 <span className="avatar">L</span><Icon name="chevron" size={14}/></button>
    </header>

    <div className="workspace">
      <aside className="rail">
        <nav>
          <button className={page === "home" ? "nav-item active" : "nav-item"} onClick={() => setPage("home")}><Icon name="home"/>概览</button>
          <button className={page === "knowledge" ? "nav-item active" : "nav-item"} onClick={() => setPage("knowledge")}><Icon name="database"/>知识库</button>
          <button className={page === "chat" ? "nav-item active" : "nav-item"} onClick={() => setPage("chat")}><Icon name="message"/>智能客服</button>
        </nav>
        <div className="rail-footer"><span className="eyebrow">STORAGE</span><span>246 chunks indexed</span><div className="usage"><i/></div><small>1.8 MB / local</small></div>
      </aside>

      {page === "home" && <section className="home-page">
        <div className="home-gridline"/><div className="home-hero"><span className="eyebrow">LOCAL RAG / CUSTOMER SUPPORT</span><h1>每一次回答，<em>都有据可循。</em></h1><p>把分散的产品资料变成可验证的客服知识。系统在回答前强制检索，并将每个结论链接回原始文档。</p><div className="hero-actions"><button className="lime-button" onClick={() => setPage("chat")}>开始问答 <Icon name="send" size={15}/></button><button className="outline-button" onClick={() => setPage("knowledge")}>管理知识库</button></div></div>
        <div className="model-console"><div className="console-head"><span><b/><b/><b/></span><small>runtime / config</small></div><div className="console-content"><p><span>01</span><strong>chat_model</strong> <i>=</i> <u>"qwen-plus"</u><mark>ready</mark></p><p><span>02</span><strong>embedding</strong> <i>=</i> <u>"text-embedding-v3"</u><mark>ready</mark></p><p><span>03</span><strong>api_key</strong> <i>=</i> <u>"••••••••••••••••"</u><mark>secured</mark></p></div></div>
      </section>}

      {page === "knowledge" && <section className="knowledge-page">
        <div className="page-heading"><div><span className="eyebrow">KNOWLEDGE MANAGEMENT</span><h1>知识库与资料</h1><p>上传 Markdown 产品资料，按标题切分并建立可追溯的检索索引。</p></div><button className="lime-button" onClick={() => setModal("create")}><Icon name="plus" size={17}/>新建知识库</button></div>
        <div className="knowledge-layout"><div className="bases-card"><div className="card-title">知识库 <span>{bases.length}</span></div>{bases.map(base => <button key={base.id} onClick={() => setSelected([base.id])} className={selected.includes(base.id) ? "base-row selected" : "base-row"}><span className="database-icon"><Icon name="database" size={16}/></span><span><b>{base.name}</b><small>{base.docs} 份文档 · {base.chunks} chunks</small></span><Icon name="more" size={17}/></button>)}</div>
        <div className="docs-panel"><div className="docs-top"><div><span className="eyebrow">ACTIVE COLLECTION</span><h2>{activeBase?.name ?? "尚未选择知识库"}</h2></div><div className="docs-actions"><button className="outline-button compact" disabled={!activeBase} onClick={() => setModal("delete")}>删除</button><button className="outline-button compact" disabled={!activeBase} onClick={() => setModal("upload")}><Icon name="upload" size={16}/>上传 Markdown</button></div></div>{activeBase ? <><div className="dropzone" onClick={() => setModal("upload")}><Icon name="upload" size={22}/><b>拖放 Markdown 文件至此处</b><span>或点击浏览 · 仅支持 UTF-8 .md</span></div><div className="doc-table"><div className="table-head"><span>文件名</span><span>状态</span><span>文本块</span><span>上传时间</span></div>{["AX3000_产品说明.md", "Mesh组网指南.md", "故障排查手册.md"].map((doc, i) => <div className="table-row" key={doc}><span><Icon name="file" size={16}/>{doc}</span><span className="completed"><i/>已完成</span><span>{[32, 18, 27][i]}</span><span>2026-08-{26 - i} 14:2{i}</span></div>)}</div></> : <div className="empty-state"><Icon name="database" size={30}/><h3>选择一个知识库</h3><p>选择后即可查看资料与上传新文档。</p></div>}</div></div>
      </section>}

      {page === "chat" && <section className="chat-page">
        <div className="chat-header"><div><span className="eyebrow">EVIDENCE-BOUND ANSWERING</span><h1>智能客服</h1></div><button className="outline-button compact" onClick={() => { setAnswerState("idle"); setQuestion("") }}>新建会话 <Icon name="plus" size={15}/></button></div>
        <div className="chat-layout"><aside className="collection-panel"><div className="panel-label"><span>检索范围</span><small>至少选择 1 个</small></div>{bases.map(base => <label className="collection-choice" key={base.id}><input type="checkbox" checked={selected.includes(base.id)} onChange={() => toggleBase(base.id)}/><span className="fake-check"><Icon name="check" size={12}/></span><span><b>{base.name}</b><small>{base.docs} docs · {base.chunks} chunks</small></span></label>)}<div className="threshold"><span>证据阈值</span><b>0.72</b><div><i/></div><small>低于阈值的片段不会用于回答</small></div></aside>
        <div className="conversation"><div className="messages"><div className="welcome"><span className="signal-icon"><i/><i/><i/></span><h2>需要查找什么资料？</h2><p>我会先在选定知识库中检索，再仅基于可靠证据回答。</p><div className="prompt-chips"><button onClick={() => setQuestion("如何恢复路由器出厂设置？")}>如何恢复出厂设置？</button><button onClick={() => setQuestion("Mesh 组网最多支持几台设备？")}>Mesh 组网数量限制</button></div></div>{answerState !== "idle" && <><div className="user-message">如何恢复路由器出厂设置？</div><div className="assistant-message"><div className="assistant-id"><span className="mini-mark">E</span> Evidence Assistant <small>刚刚</small></div>{answerState === "thinking" ? <div className="typing"><i/><i/><i/> 正在检索已选知识库…</div> : answerState === "refused" ? <p>当前选定资料中没有足够证据支持这个问题的回答。请补充相关产品资料，或调整检索范围后再试。</p> : <p>请在设备通电状态下，使用针状物长按机身背面的 <strong>Reset</strong> 按钮约 8 秒，待状态指示灯快速闪烁后松开。设备重启完成后将恢复出厂默认设置，原有 Wi‑Fi 名称、密码和上网配置会被清除。<sup>[1]</sup></p>} {answerState !== "thinking" && <div className="answer-footer"><span>{answerState === "refused" ? "证据不足 · 已拒答" : "回答完成 · 1 条引用"}</span><button>复制回答</button></div>}</div></>}</div><form className="composer" onSubmit={ask}><textarea value={question} onChange={e => setQuestion(e.target.value)} placeholder={selected.length ? "输入客户问题…" : "请先在左侧选择知识库"} disabled={!selected.length}/><div><span><kbd>⌘</kbd><kbd>↵</kbd> 发送</span><button aria-label="发送问题" disabled={!selected.length}><Icon name="send" size={18}/></button></div></form></div>
        <aside className="evidence-panel"><div className="evidence-title"><span>执行过程</span><small>本次会话</small></div><div className="timeline"><div className="event done"><i/><span>问题改写完成</span><small>14:32:08</small></div><div className="event done"><i/><span>在 1 个知识库中检索</span><small>14:32:09</small></div><div className="event done"><i/><span>命中 3 条候选片段</span><small>14:32:09</small></div><div className="event done"><i/><span>证据判定：充分</span><small>14:32:10</small></div></div><div className="evidence-title source-head"><span>引用证据</span><small>1 / 3</small></div><article className="evidence-card"><div className="evidence-meta"><span className="evidence-num">1</span><b>0.91</b><span>高相关</span></div><h3>AX3000_产品说明.md</h3><p className="path">使用与维护 / 恢复出厂设置</p><blockquote>“长按 Reset 按钮 8 秒，待指示灯快速闪烁后松开。恢复出厂设置将清除所有自定义网络配置。”</blockquote><button>查看原文 <Icon name="chevron" size={14}/></button></article></aside></div>
      </section>}
    </div>
    {modal && <div className="modal-backdrop" onMouseDown={() => setModal(null)}><div className="modal" onMouseDown={e => e.stopPropagation()}><button className="modal-close" onClick={() => setModal(null)}><Icon name="close"/></button>{modal === "create" ? <><span className="eyebrow">NEW COLLECTION</span><h2>创建知识库</h2><p>知识库用于限制检索范围，名称创建后仍可在此处管理。</p><label className="field-label">知识库名称<input autoFocus value={newBase} onChange={e => setNewBase(e.target.value)} placeholder="例如：智能家居产品资料"/></label><button className="lime-button full" onClick={createBase}>创建知识库</button></> : modal === "upload" ? <><span className="eyebrow">ADD DOCUMENTS</span><h2>上传 Markdown</h2><div className="modal-upload"><Icon name="upload" size={28}/><b>选择 .md 文件</b><small>支持一次上传多份 UTF-8 Markdown 文档</small></div><button className="lime-button full" onClick={() => { setModal(null); notify("文件校验与索引任务已开始") }}>选择文件</button></> : <><span className="eyebrow">CONFIRM REMOVAL</span><h2>删除知识库？</h2><p>将永久删除 <b>{activeBase?.name}</b> 中的文档、元数据及全部向量块。此操作无法撤销。</p><div className="modal-actions"><button className="outline-button" onClick={() => setModal(null)}>取消</button><button className="danger-button" onClick={deleteBase}>确认删除</button></div></>}</div></div>}
    {toast && <div className="toast"><span className="status-dot"/>{toast}</div>}
  </main>
}
