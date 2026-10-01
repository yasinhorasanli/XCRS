import type {
  Board,
  KnowledgeUnit,
  KnowledgeUnitGroup,
  RecommendationResponse,
  RelatedUnit,
} from '~/types/api'

/** Calls to the backend's /api/v1 (proxied by the Nuxt server). */
export function useXcrsApi() {
  return {
    recommend: (board: Board) =>
      $fetch<RecommendationResponse>('/api/v1/recommendations', { method: 'POST', body: board }),

    recommendation: (id: string) => $fetch<RecommendationResponse>(`/api/v1/recommendations/${id}`),

    feedback: (id: string, body: { role_id?: number; course_id?: number; rating: 1 | -1 }) =>
      $fetch(`/api/v1/recommendations/${id}/feedback`, { method: 'POST', body }),

    groups: () => $fetch<{ groups: KnowledgeUnitGroup[] }>('/api/v1/knowledge-units/groups'),

    search: (q: string, limit = 8) =>
      $fetch<{ units: KnowledgeUnit[] }>('/api/v1/knowledge-units', { query: { q, limit } }),

    // phrases: enjoyed + curious (what to suggest near); avoid: didn't enjoy; exclude: anything else entered
    related: (body: { phrases: string[]; avoid: string[]; exclude: string[] }, limit = 14) =>
      $fetch<{ units: RelatedUnit[] }>('/api/v1/knowledge-units/related', {
        method: 'POST',
        body: { ...body, limit },
      }),
  }
}

/** Drag-and-drop payload between the suggestion panel and the buckets. */
export const DRAG_TYPE = 'application/x-xcrs-skill'
