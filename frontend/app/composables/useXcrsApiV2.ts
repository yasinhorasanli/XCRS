import type {
  ChipInput,
  CvImportStatus,
  Experience,
  Me,
  PhraseMatch,
  Provider,
  RecommendationV2,
  ResultSummary,
  SavedBoard,
  SkillGroup,
  SkillSuggestion,
} from '~/types/apiV2'

/** Calls to the backend's /api/v2 (engine v2), proxied by the Nuxt server. During server rendering the browser's
 * cookies are passed on (useRequestFetch), so the proxy knows who is signed in (ADR-0043). */
export function useXcrsApiV2() {
  const $fetch = useRequestFetch()
  return {
    searchSkills: (q: string, limit = 8) =>
      $fetch<{ skills: SkillSuggestion[] }>('/api/v2/skills', { query: { q, limit } }),

    groups: () => $fetch<{ groups: SkillGroup[] }>('/api/v2/skills/groups'),

    match: (phrases: string[]) =>
      $fetch<{ matches: PhraseMatch[] }>('/api/v2/skills/match', { method: 'POST', body: { phrases } }),

    recommend: (chips: ChipInput[], experience?: Experience | null) =>
      $fetch<RecommendationV2>('/api/v2/recommendations', { method: 'POST', body: { chips, ...(experience ? { experience } : {}) } }),

    recommendation: (id: string) => $fetch<RecommendationV2>(`/api/v2/recommendations/${id}`),

    feedback: (id: string, body: { role?: string; rating: 1 | -1 }) =>
      $fetch(`/api/v2/recommendations/${id}/feedback`, { method: 'POST', body }),

    // Accounts (ADR-0043); all need a signed-in user except `providers`.
    providers: () => $fetch<{ providers: Provider[]; dev: boolean }>('/api/auth/providers'),
    me: () => $fetch<Me>('/api/v2/me'),
    board: () => $fetch<{ board: SavedBoard | null }>('/api/v2/me/board'),
    saveBoard: (chips: ChipInput[], experience?: Experience | null) =>
      $fetch('/api/v2/me/board', { method: 'PUT', body: { chips, ...(experience ? { experience } : {}) } }),
    results: () => $fetch<ResultSummary[]>('/api/v2/me/results'),
    saveResult: (id: string) => $fetch<ResultSummary>(`/api/v2/me/results/${id}`, { method: 'POST' }),
    deleteResult: (id: string) => $fetch(`/api/v2/me/results/${id}`, { method: 'DELETE' }),
    signOutEverywhere: () => $fetch('/api/v2/me/sign-out-everywhere', { method: 'POST' }),
    deleteAccount: () => $fetch('/api/v2/me', { method: 'DELETE' }),

    // CV import (ADR-0045), signed in only: a PDF (field "file") or pasted text (field "text").
    cvImport: (body: FormData) => $fetch<CvImportStatus>('/api/v2/cv-imports', { method: 'POST', body }),
    cvImportStatus: (id: string) => $fetch<CvImportStatus>(`/api/v2/cv-imports/${id}`),
  }
}

export const LEVEL_NAMES: Record<string, string> = {
  entry: 'Entry',
  mid: 'Mid-level',
  senior: 'Senior',
  staff: 'Staff / Principal',
}

export const PROFICIENCY_NAMES = ['', 'Basic', 'Working', 'Advanced', 'Expert'] as const

export const EXPERIENCE_OPTIONS: { value: Experience; label: string }[] = [
  { value: 'student', label: 'Student' },
  { value: '0-2', label: 'Under 2 years' },
  { value: '2-5', label: '2–5 years' },
  { value: '5-10', label: '5–10 years' },
  { value: '10+', label: '10+ years' },
]

export const PROVIDER_META: Record<Provider, { label: string; icon: string }> = {
  github: { label: 'GitHub', icon: 'i-simple-icons-github' },
  google: { label: 'Google', icon: 'i-simple-icons-google' },
  linkedin: { label: 'LinkedIn', icon: 'i-simple-icons-linkedin' },
}
