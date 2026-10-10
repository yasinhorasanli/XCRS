import { CATEGORY_META } from '~/composables/categories'
import type { Category } from '~/types/apiV2'
import type { ChipInput, Experience, RecommendationV2, SkillRef } from '~/types/apiV2'

export const MAX_CHIPS = 60 // the API's limit

export interface Chip {
  key: string // lower-case label, unique on the board
  label: string
  category: Category
  skill?: string // picked from the catalog
  proficiency?: number // 1-4, optional
  match: { status: 'picked' | 'pending' | 'done' | 'error'; skills: SkillRef[]; method?: string }
}

export const EXAMPLE_CHIPS: { label: string; category: Category; skill?: string; proficiency?: number }[] = [
  { label: 'Python', category: 'liked', skill: 'python', proficiency: 3 },
  { label: 'building REST APIs with Django', category: 'liked', proficiency: 2 },
  { label: 'PostgreSQL', category: 'liked', skill: 'postgresql', proficiency: 2 },
  { label: 'Docker', category: 'neutral', skill: 'docker', proficiency: 2 },
  { label: 'CSS', category: 'disliked', skill: 'css', proficiency: 1 },
  { label: 'Kubernetes', category: 'curious', skill: 'kubernetes' },
  { label: 'data pipelines', category: 'curious' },
]

/**
 * The engine v2 board (ADR-0029): chips are catalog skills (picked) or typed text. Typed chips are
 * matched in the background as soon as they are added (ADR-0030), so "Recommend" rarely waits.
 */
export function useBoardV2() {
  const chips = useState<Chip[]>('board-v2', () => [])
  const active = useState<Category>('board-v2-active', () => 'liked')
  const experience = useState<Experience | null>('board-v2-experience', () => null) // optional (ADR-0041)
  const api = useXcrsApiV2()
  const toast = useToast()

  const total = computed(() => chips.value.length)
  const pending = computed(() => chips.value.filter((c) => c.match.status === 'pending').length)
  const byCategory = (category: Category) => chips.value.filter((c) => c.category === category)

  function find(label: string) {
    return chips.value.find((c) => c.key === label.trim().toLowerCase())
  }

  /** The catalog skills a chip stands for: the picked one, or what its text was matched to. */
  const skillsOf = (c: Chip) => (c.skill ? [c.skill] : c.match.skills.map((s) => s.id))
  const sameSkills = (a: string[], b: string[]) => a.length > 0 && a.length === b.length && a.every((s) => b.includes(s))

  function announceMove(chip: Chip, from: Category, why = '') {
    toast.add({
      title: `Moved “${chip.label}”`,
      description: `${why}From “${CATEGORY_META[from].title}” to “${CATEGORY_META[chip.category].title}”: a skill can be in one box only.`,
      actions: [{ label: 'Undo', color: 'neutral', variant: 'outline', onClick: () => { chip.category = from } }],
    })
  }

  /** A typed chip that turns out to be a skill already on the board ("dockerrr" after "Docker") is merged
   * into the existing chip: the newest box wins, as when the same skill is added twice. */
  function mergeDuplicate(chip: Chip) {
    const skills = skillsOf(chip)
    const other = chips.value.find((c) => c.key !== chip.key && sameSkills(skillsOf(c), skills))
    if (!other) return
    chips.value = chips.value.filter((c) => c.key !== chip.key)
    if (chip.proficiency && !other.proficiency) other.proficiency = chip.proficiency
    const reads = `“${chip.label}” reads as ${chip.match.skills.map((s) => s.name).join(', ')}. `
    if (other.category === chip.category) {
      toast.add({ title: `Already on the board`, description: `${reads}It is already in “${CATEGORY_META[other.category].title}” as “${other.label}”.` })
    } else {
      const from = other.category
      other.category = chip.category
      announceMove(other, from, reads)
    }
  }

  async function matchInBackground(key: string) {
    const chip = chips.value.find((c) => c.key === key)
    if (!chip || chip.skill) return
    try {
      const { matches } = await api.match([chip.label])
      const current = chips.value.find((c) => c.key === key)
      if (current && !current.skill) {
        current.match = { status: 'done', skills: matches[0]?.skills ?? [], method: matches[0]?.method }
        mergeDuplicate(current)
      }
    } catch {
      const current = chips.value.find((c) => c.key === key)
      if (current) current.match = { status: 'error', skills: [] } // the server matches again on submit
    }
  }

  /** Add a picked skill or typed text. A skill has one feeling, so an existing chip moves to the new category;
   * the move is announced, with Undo, so it never happens silently. `matched`: typed text already matched (a
   * restored board matches all its phrases in one request), so no background request is made. */
  function add(
    label: string,
    category: Category = active.value,
    skill?: SkillRef,
    proficiency?: number,
    matched?: { skills: SkillRef[]; method?: string },
  ) {
    const clean = label.replace(/\s+/g, ' ').trim().slice(0, 100)
    if (!clean) return 'empty' as const
    const existing = find(clean) ?? (skill ? chips.value.find((c) => sameSkills(skillsOf(c), [skill.id])) : undefined)
    if (existing) {
      if (existing.category === category) return 'exists' as const
      const from = existing.category
      existing.category = category
      announceMove(existing, from)
      return 'moved' as const
    }
    if (chips.value.length >= MAX_CHIPS) return 'full' as const
    const chip: Chip = {
      key: clean.toLowerCase(),
      label: clean,
      category,
      skill: skill?.id,
      proficiency,
      match: skill
        ? { status: 'picked', skills: [skill] }
        : matched
          ? { status: 'done', skills: matched.skills, method: matched.method }
          : { status: 'pending', skills: [] },
    }
    chips.value = [...chips.value, chip]
    if (!skill && !matched) matchInBackground(chip.key)
    return 'added' as const
  }

  function remove(key: string) {
    chips.value = chips.value.filter((c) => c.key !== key)
  }

  function rate(key: string, proficiency: number | undefined) {
    const chip = chips.value.find((c) => c.key === key)
    if (chip) chip.proficiency = chip.proficiency === proficiency ? undefined : proficiency
  }

  function clear() {
    chips.value = []
    experience.value = null
  }

  function fillExample() {
    clear()
    for (const c of EXAMPLE_CHIPS) add(c.label, c.category, c.skill ? { id: c.skill, name: c.label } : undefined, c.proficiency)
  }

  const asInput = (): ChipInput[] =>
    chips.value.map((c) => ({
      category: c.category,
      ...(c.skill ? { skill: c.skill } : { text: c.label }),
      ...(c.proficiency ? { proficiency: c.proficiency } : {}),
    }))

  /** The board a result was made from (also for a shared link); typed chips keep their matches, so no request. */
  function fromResult(result: RecommendationV2) {
    clear()
    for (const m of result.matched) {
      const picked = m.method === 'picked' ? m.skills[0] : undefined
      const matched = picked ? undefined : { skills: m.skills, method: m.method }
      add(m.text ?? picked?.name ?? '', m.category, picked, m.proficiency ?? undefined, matched)
    }
    experience.value = result.experience ?? null
  }

  /** Whether a catalog skill is already on the board (picked, matched from typed text, or a chip's label). */
  const onBoard = (skill: { id: string; name: string }) =>
    chips.value.some((c) => skillsOf(c).includes(skill.id) || c.key === skill.name.trim().toLowerCase())

  return { chips, active, experience, total, pending, byCategory, add, remove, rate, clear, fillExample, fromResult, asInput, onBoard }
}
