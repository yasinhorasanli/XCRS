import type { ChipInput, PhraseMatch, RecommendationV2, SkillGroup, SkillSuggestion } from '~/types/apiV2'

/** Calls to the backend's /api/v2 (engine v2), proxied by the Nuxt server. */
export function useXcrsApiV2() {
  return {
    searchSkills: (q: string, limit = 8) =>
      $fetch<{ skills: SkillSuggestion[] }>('/api/v2/skills', { query: { q, limit } }),

    groups: () => $fetch<{ groups: SkillGroup[] }>('/api/v2/skills/groups'),

    match: (phrases: string[]) =>
      $fetch<{ matches: PhraseMatch[] }>('/api/v2/skills/match', { method: 'POST', body: { phrases } }),

    recommend: (chips: ChipInput[]) =>
      $fetch<RecommendationV2>('/api/v2/recommendations', { method: 'POST', body: { chips } }),

    recommendation: (id: string) => $fetch<RecommendationV2>(`/api/v2/recommendations/${id}`),

    feedback: (id: string, body: { role?: string; rating: 1 | -1 }) =>
      $fetch(`/api/v2/recommendations/${id}/feedback`, { method: 'POST', body }),
  }
}

export const LEVEL_NAMES: Record<string, string> = {
  entry: 'Entry',
  mid: 'Mid-level',
  senior: 'Senior',
  staff: 'Staff / Principal',
}

export const PROFICIENCY_NAMES = ['', 'Basic', 'Working', 'Advanced', 'Expert'] as const
