export interface SearchResult {
  payload: Record<string, any>
  score?: number
}

export interface VectorSearchRequest {
  vector: number[]
  vector_name: string
  top_k?: number
  filters?: Record<string, string[]>
}

export interface HybridSearchRequest {
  vectors: Record<string, number[]>
  top_k?: number
  filters?: Record<string, string[]>
}

export interface ChatRequest {
  session_id: string
  message: string
}
