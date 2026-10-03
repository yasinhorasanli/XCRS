// The /api/v2 contract (engine v2 on the new catalog: ADR-0029, ADR-0030, ADR-0031).

export type Category = 'liked' | 'neutral' | 'disliked' | 'curious'

export interface SkillRef {
  id: string
  name: string
}

export interface SkillSuggestion extends SkillRef {
  kind: string
}

export interface SkillGroup {
  family: string
  name: string
  skills: SkillSuggestion[]
}

export type MatchMethod = 'lookup' | 'llm' | 'cache' | 'embedding' | 'none'

export interface PhraseMatch {
  phrase: string
  skills: SkillRef[]
  method: MatchMethod
}

export interface ChipInput {
  category: Category
  skill?: string
  text?: string
  proficiency?: number
}

export interface Level {
  id: string
  title: string | null
  coverage: number
}

export interface Gap {
  skills: SkillRef[] // several = any one of them
  need: number
  have: number
  stage: string
}

export interface Resource {
  id: string
  title: string
  url: string
  provider: string
  type: string
  level: string | null
  free: boolean
  curated: boolean
  skills: SkillRef[] // the role's gaps it covers
}

export interface RoleResult {
  id: string
  name: string
  family: string
  score: number
  interest: number
  coverage: number
  level: Level | null
  target_level: Level
  levels: Level[]
  because: (SkillRef & { category: Category })[]
  gaps: Gap[] // what the next level adds (ADR-0041)
  gaps_total: number
  basics?: Gap[] // unlisted skills of the levels reached: assumed, to check
  resources: Resource[]
  explanation_status: 'pending' | 'done' | 'failed' | 'disabled'
  explanation: string | null // written by the local LLM in the background
  next_step: string | null
}

export interface MatchedChip {
  text: string | null
  category: Category
  proficiency: number | null
  method: string
  skills: SkillRef[]
}

export interface RecommendationV2 {
  id: string
  created_at: string
  status: 'ok' | 'insufficient_input'
  algorithm_version: string
  catalog_version: string
  matched: MatchedChip[]
  roles: RoleResult[]
  experience?: Experience | null
}

/** Years in software, optional on the board (ADR-0041). */
export type Experience = 'student' | '0-2' | '2-5' | '5-10' | '10+'
