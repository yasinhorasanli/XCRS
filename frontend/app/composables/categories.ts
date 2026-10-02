import type { Category } from '~/types/apiV2'

// The four board categories (ADR-0023).
export const CATEGORIES: Category[] = ['liked', 'neutral', 'disliked', 'curious']

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
