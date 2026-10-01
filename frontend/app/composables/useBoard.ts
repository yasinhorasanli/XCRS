import type { Board, Category } from '~/types/api'

export const CATEGORIES: Category[] = ['liked', 'neutral', 'disliked', 'curious']
export const MAX_PER_CATEGORY = 30 // the API's limit (ADR-0016)
export const MAX_LENGTH = 100

// Static class strings (Tailwind only keeps classes it can see in the source).
export const CATEGORY_META: Record<Category, {
  title: string
  short: string
  hint: string
  icon: string
  chip: string
  ring: string
  dot: string
  soft: string
  activeTab: string
}> = {
  liked: {
    short: 'Enjoyed',
    title: 'I enjoyed',
    hint: 'Courses, subjects or tools you studied and liked',
    icon: 'i-heroicons-heart',
    chip: 'bg-emerald-50 text-emerald-800 ring-emerald-200',
    ring: 'ring-emerald-400',
    dot: 'bg-emerald-500',
    soft: 'bg-emerald-50/60',
    activeTab: 'bg-emerald-600 text-white',
  },
  neutral: {
    short: 'Neutral',
    title: 'Neutral about',
    hint: 'Things you studied without strong feelings',
    icon: 'i-heroicons-minus-circle',
    chip: 'bg-slate-100 text-slate-700 ring-slate-200',
    ring: 'ring-slate-400',
    dot: 'bg-slate-400',
    soft: 'bg-slate-100/60',
    activeTab: 'bg-slate-600 text-white',
  },
  disliked: {
    short: 'Disliked',
    title: "Didn't enjoy",
    hint: 'Things you studied and would rather avoid',
    icon: 'i-heroicons-hand-thumb-down',
    chip: 'bg-rose-50 text-rose-800 ring-rose-200',
    ring: 'ring-rose-400',
    dot: 'bg-rose-500',
    soft: 'bg-rose-50/60',
    activeTab: 'bg-rose-600 text-white',
  },
  curious: {
    short: 'Curious',
    title: 'Curious about',
    hint: 'Things you want to learn next',
    icon: 'i-heroicons-light-bulb',
    chip: 'bg-violet-50 text-violet-800 ring-violet-200',
    ring: 'ring-violet-400',
    dot: 'bg-violet-500',
    soft: 'bg-violet-50/60',
    activeTab: 'bg-violet-600 text-white',
  },
}

export const EXAMPLE_BOARD: Board = {
  liked: ['Java', 'SQL', 'Spring Boot'],
  neutral: ['HTML'],
  disliked: ['PHP'],
  curious: ['Docker', 'Kubernetes'],
}

const emptyBoard = (): Board => ({ liked: [], neutral: [], disliked: [], curious: [] })

/** The skill board, shared between the input page and the results page ("Edit my answers"). */
export function useBoard() {
  const board = useState<Board>('board', emptyBoard)
  const active = useState<Category>('active-category', () => 'liked')

  const all = computed(() => CATEGORIES.flatMap((c) => board.value[c]))
  const total = computed(() => all.value.length)

  function categoryOf(label: string): Category | undefined {
    const key = label.toLowerCase()
    return CATEGORIES.find((c) => board.value[c].some((l) => l.toLowerCase() === key))
  }

  /** Put a skill into a category. A skill lives in one category only, so this also moves it. */
  function add(label: string, category: Category = active.value): 'added' | 'moved' | 'exists' | 'full' | 'empty' {
    const clean = label.replace(/\s+/g, ' ').trim().slice(0, MAX_LENGTH)
    if (!clean) return 'empty'
    const current = categoryOf(clean)
    if (current === category) return 'exists'
    if (board.value[category].length >= MAX_PER_CATEGORY) return 'full'
    if (current) remove(clean)
    board.value[category] = [...board.value[category], clean]
    return current ? 'moved' : 'added'
  }

  function remove(label: string) {
    const key = label.toLowerCase()
    for (const c of CATEGORIES) {
      board.value[c] = board.value[c].filter((l) => l.toLowerCase() !== key)
    }
  }

  function clear() {
    board.value = emptyBoard()
  }

  function fill(source: Partial<Board>) {
    board.value = { ...emptyBoard(), ...structuredClone(toRaw(source)) }
  }

  return { board, active, all, total, add, remove, clear, fill, categoryOf }
}
