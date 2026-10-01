// The v1 API contract (ADR-0016), as served by the backend through /api/v1/**.

export type Category = 'liked' | 'neutral' | 'disliked' | 'curious'

export type ExplanationStatus = 'pending' | 'done' | 'failed' | 'disabled'

export interface Course {
  course_id: number
  title: string
  url: string
  explanation: string | null
  concepts: string[]
  similarity: number
}

export interface Role {
  role_id: number
  role: string
  score: number
  explanation: string | null
  explanation_status: ExplanationStatus
  next_to_learn: string[]
  courses: Course[]
}

export interface RecommendationResponse {
  request_id: string
  status: 'ok' | 'insufficient_input'
  model: string
  latency_ms: number
  input: Partial<Record<Category, string[]>>
  roles: Role[]
}

export interface KnowledgeUnitGroup {
  name: string
  units: string[]
}

export interface KnowledgeUnit {
  label: string
  source: 'curated' | 'roadmap'
  roles: string[]
}

export interface RelatedUnit {
  label: string
  because: string
  similarity: number
}

export type Board = Record<Category, string[]>
