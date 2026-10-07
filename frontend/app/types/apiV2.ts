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

export interface ResourceSection {
  title: string
  url: string
  start_seconds: number | null
  kind: 'chapter' | 'video' | 'start' // a long video's chapter, a playlist's video, or episode 1
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
  duration_minutes?: number | null
  section?: ResourceSection | null // the part to open for these gaps (ADR-0046)
}

export interface RoleResult {
  id: string
  name: string
  family: string
  title_for_you?: string | null // from the learner's strongest title skills (ADR-0044)
  job_titles?: string[] // the role's name and other market titles
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
  saved?: boolean // in the signed-in viewer's account (ADR-0043)
  can_save?: boolean // anonymous and recent: "Save to my account" can attach it
}

/** A skill added to the board from a result (gaps and assumed basics). */
export interface SkillToAdd {
  skill: SkillRef
  category: Category
  proficiency?: number
}

/** Years in software, optional on the board (ADR-0041). */
export type Experience = 'student' | '0-2' | '2-5' | '5-10' | '10+'

/** Accounts (ADR-0043). */
export type Provider = 'github' | 'google' | 'linkedin'

export interface Me {
  id: string
  display_name: string | null
  email: string | null
  providers: Provider[]
  created_at: string
}

export interface SavedBoard {
  chips: (ChipInput & { name: string | null })[]
  experience?: Experience | null
  updated_at: string
}

export interface ResultSummary {
  id: string
  created_at: string
  status: 'ok' | 'insufficient_input'
  roles: { id: string; name: string; level: string | null }[]
}

// CV import (ADR-0045)
export interface CvSuggestion {
  skill: string
  name: string
  proficiency: number | null // suggested 1-3 from years of use
  years: number | null
  evidence: string // a short quote from the CV: render as text
  job: number | null // index into `jobs` of the newest job it was used in; null = summary or skills list
  source: 'llm' | 'scan'
}

export interface CvJob {
  title: string
  employer: string
  start: string | null
  end: string | null
  kind: string
}

export interface CvImportResult {
  suggestions: CvSuggestion[]
  experience: Experience | null
  jobs: CvJob[]
  warnings: { hidden: number; instructions: number; examples: string[] }
}

export interface CvImportStatus {
  id: string
  status: 'queued' | 'running' | 'done' | 'failed'
  position: number
  result?: CvImportResult | null
}
