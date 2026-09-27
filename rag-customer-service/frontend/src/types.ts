export type SystemStatus = {
  status: "ready"
  chat_model: string
  embedding_model: string
  retrieval_score_threshold: number
  rerank_model?: string
  retrieval_top_k?: number
  knowledge_base_count: number
  document_count: number
  chunk_count: number
}

export type KnowledgeBaseSummary = {
  id: string
  name: string
  created_at: string
  document_count: number
  chunk_count: number
}

export type DocumentSummary = {
  id: string
  kb_id: string
  filename: string
  chunk_count: number
  status: string
  created_at: string
  progress?: string[]
}

export type UploadBatchResult = {
  documents: DocumentSummary[]
  errors: Array<{
    filename: string
    detail: string
  }>
}

export type ChatHistoryMessage = {
  role: "user" | "assistant"
  content: string
}

export type EvidenceSummary = {
  reference_id: number
  subquestion_id: string
  subquestion: string
  support_status: "supporting" | "related"
  kb_id: string
  document_id: string
  document_name: string
  heading_path: string[]
  chunk_index: number
  content: string
  distance: number
  similarity: number
  rerank_score?: number | null
}

export type TraceEvent = {
  type: "node_status" | "tool_status" | "evidence" | "answer_delta" | "completed" | "error"
  stage: string
  message: string
  payload: {
    count?: number
    verified_count?: number
    kb_ids?: string[]
    reason?: string
    result?: {
      query: string
      evidences: EvidenceSummary[]
      has_sufficient_evidence: boolean
      reason: string
    }
  }
}
