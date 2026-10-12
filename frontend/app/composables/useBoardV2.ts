import { CATEGORIES, CATEGORY_META } from '~/composables/categories'
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

type ExampleChip = { label: string; category: Category; skill?: string; proficiency?: number }

/** "Try an example" boards: a working developer, and a computer science / engineering student who has taken the
 * core courses but none of the electives yet (2026-10-12). */
export const EXAMPLES: { id: string; label: string; experience: Experience | null; chips: ExampleChip[] }[] = [
  {
    id: 'developer',
    label: 'A backend developer',
    experience: null,
    chips: [
      { label: 'Python', category: 'liked', skill: 'python', proficiency: 3 },
      { label: 'building REST APIs with Django', category: 'liked', proficiency: 2 },
      { label: 'PostgreSQL', category: 'liked', skill: 'postgresql', proficiency: 2 },
      { label: 'Docker', category: 'neutral', skill: 'docker', proficiency: 2 },
      { label: 'CSS', category: 'disliked', skill: 'css', proficiency: 1 },
      { label: 'Kubernetes', category: 'curious', skill: 'kubernetes' },
      { label: 'data pipelines', category: 'curious' },
    ],
  },
  {
    id: 'cs-student',
    label: 'A computer science student',
    experience: 'student',
    chips: [
      // Core courses taken
      { label: 'Programming fundamentals', category: 'liked', skill: 'programming-fundamentals', proficiency: 2 },
      { label: 'Data structures', category: 'liked', skill: 'data-structures', proficiency: 2 },
      { label: 'Algorithms and complexity', category: 'liked', skill: 'algorithms', proficiency: 2 },
      { label: 'Object-oriented programming', category: 'liked', skill: 'oop', proficiency: 2 },
      { label: 'Java', category: 'liked', skill: 'java', proficiency: 2 },
      { label: 'Python', category: 'liked', skill: 'python', proficiency: 2 },
      { label: 'C', category: 'neutral', skill: 'c', proficiency: 2 },
      { label: 'SQL', category: 'neutral', skill: 'sql', proficiency: 2 },
      { label: 'Relational databases', category: 'neutral', skill: 'relational-databases', proficiency: 2 },
      { label: 'Operating systems', category: 'neutral', skill: 'operating-systems', proficiency: 2 },
      { label: 'Computer architecture', category: 'neutral', skill: 'computer-architecture', proficiency: 1 },
      { label: 'Git', category: 'neutral', skill: 'git', proficiency: 1 },
      { label: 'Linux and the command line', category: 'neutral', skill: 'linux', proficiency: 1 },
      { label: 'Testing fundamentals', category: 'neutral', skill: 'testing-fundamentals', proficiency: 1 },
      { label: 'Statistics and probability', category: 'neutral', skill: 'statistics', proficiency: 1 },
      { label: 'Digital electronics', category: 'disliked', skill: 'digital-electronics', proficiency: 1 },
      { label: 'Calculus', category: 'disliked', skill: 'calculus', proficiency: 2 },
      // Electives not taken yet
      { label: 'Machine learning fundamentals', category: 'curious', skill: 'machine-learning-fundamentals' },
      { label: 'Computer networking', category: 'curious', skill: 'computer-networking' },
      { label: 'Computer graphics and rendering', category: 'curious', skill: 'computer-graphics' },
      { label: 'LLM fundamentals', category: 'curious', skill: 'llm-fundamentals' },
      { label: 'Quantum computing', category: 'curious', skill: 'quantum-computing' },
    ],
  },
]

/**
 * The engine v2 board (ADR-0029): chips are catalog skills (picked) or typed text. Typed chips are
 * matched in the background as soon as they are added (ADR-0030), so "Recommend" rarely waits.
 */
export function useBoardV2() {
  const chips = useState<Chip[]>('board-v2', () => [])
  const active = useState<Category>('board-v2-active', () => 'liked')
  const experience = useState<Experience | null>('board-v2-experience', () => null) // optional (ADR-0041)
  // Attract mode (2026-10-12): on the home page the selected box moves with the headline's flips until the visitor
  // does anything on the board; it starts again on the next visit or reload.
  const attract = useState('board-attract', () => false)
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

  /** Starts with an empty board on the home page (every visit), unless the visitor prefers reduced motion. */
  function startAttract() {
    if (!chips.value.length && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) attract.value = true
  }
  /** Any interaction with the board ends it for this visit. */
  function stopAttract() {
    attract.value = false
  }
  /** One step: the next box, in board order. */
  function attractStep() {
    if (attract.value) active.value = CATEGORIES[(CATEGORIES.indexOf(active.value) + 1) % CATEGORIES.length]!
  }

  function fillExample(id: string = EXAMPLES[0]!.id) {
    const example = EXAMPLES.find((e) => e.id === id) ?? EXAMPLES[0]!
    clear()
    for (const c of example.chips) add(c.label, c.category, c.skill ? { id: c.skill, name: c.label } : undefined, c.proficiency)
    experience.value = example.experience
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

  return {
    chips,
    active,
    experience,
    total,
    pending,
    byCategory,
    add,
    remove,
    rate,
    clear,
    fillExample,
    fromResult,
    asInput,
    onBoard,
    attract,
    startAttract,
    stopAttract,
    attractStep,
  }
}
