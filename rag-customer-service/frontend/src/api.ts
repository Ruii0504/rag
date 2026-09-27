import type {
  ChatHistoryMessage,
  DocumentSummary,
  KnowledgeBaseSummary,
  SystemStatus,
  TraceEvent,
  UploadBatchResult,
} from "./types"


async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail ?? "请求失败，请稍后重试")
  }
  return response.json() as Promise<T>
}

export const loadStatus = () => requestJson<SystemStatus>("/api/status")

export const listKnowledgeBases = () => requestJson<KnowledgeBaseSummary[]>(
  "/api/knowledge-bases",
)

export const listDocuments = (kbId: string) => requestJson<DocumentSummary[]>(
  `/api/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
)

export const createKnowledgeBase = (name: string) => requestJson<KnowledgeBaseSummary>(
  "/api/knowledge-bases",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  },
)

export async function deleteKnowledgeBase(kbId: string): Promise<void> {
  const response = await fetch(`/api/knowledge-bases/${encodeURIComponent(kbId)}`, {
    method: "DELETE",
  })
  if (!response.ok) throw new Error("知识库删除失败")
}

export async function deleteDocument(kbId: string, documentId: string): Promise<void> {
  const response = await fetch(
    `/api/knowledge-bases/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" },
  )
  if (!response.ok) throw new Error("文档删除失败")
}

export function uploadDocuments(kbId: string, files: File[]) {
  const body = new FormData()
  files.forEach(file => body.append("files", file))
  return requestJson<UploadBatchResult>(
    `/api/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
    { method: "POST", body },
  )
}

export async function streamChat(
  request: {
    question: string
    history: ChatHistoryMessage[]
    kb_ids: string[]
  },
  onEvent: (event: TraceEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  })
  if (!response.ok) throw new Error("客服请求失败，请稍后重试")
  if (!response.body) throw new Error("浏览器未收到客服事件流")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  const consume = (complete: boolean) => {
    buffer += complete ? decoder.decode() : ""
    const blocks = buffer.split("\n\n")
    buffer = complete ? "" : (blocks.pop() ?? "")
    for (const block of blocks) {
      const data = block.split("\n")
        .find(line => line.startsWith("data: "))
        ?.slice(6)
      if (data) onEvent(JSON.parse(data) as TraceEvent)
    }
    if (complete && buffer.trim()) {
      const data = buffer.trim().replace(/^data: /, "")
      if (data) onEvent(JSON.parse(data) as TraceEvent)
      buffer = ""
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      consume(true)
      break
    }
    buffer += decoder.decode(value, { stream: true })
    consume(false)
  }
}
