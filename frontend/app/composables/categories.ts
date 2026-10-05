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
  text: string
  border: string
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
    chip: 'bg-emerald-50 text-emerald-800 ring-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-200 dark:ring-emerald-800',
    text: 'text-emerald-800 dark:text-emerald-200',
    border: 'ring-emerald-200 dark:ring-emerald-800',
    ring: 'ring-emerald-400 dark:ring-emerald-400',
    dot: 'bg-emerald-500 dark:bg-emerald-400',
    soft: 'bg-emerald-50/60 dark:bg-emerald-950/40',
    activeTab: 'bg-emerald-700 text-white dark:bg-emerald-700 dark:text-white',
  },
  neutral: {
    short: 'Neutral',
    title: 'Neutral about',
    hint: 'Things you studied without strong feelings',
    icon: 'i-heroicons-minus-circle',
    chip: 'bg-slate-100 text-slate-700 ring-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:ring-slate-700',
    text: 'text-slate-700 dark:text-slate-200',
    border: 'ring-slate-200 dark:ring-slate-700',
    ring: 'ring-slate-400 dark:ring-slate-400',
    dot: 'bg-slate-400 dark:bg-slate-400',
    soft: 'bg-slate-100/60 dark:bg-slate-800',
    activeTab: 'bg-slate-600 text-white dark:bg-slate-700 dark:text-white',
  },
  disliked: {
    short: 'Disliked',
    title: "Didn't enjoy",
    hint: 'Things you studied and would rather avoid',
    icon: 'i-heroicons-hand-thumb-down',
    chip: 'bg-rose-50 text-rose-800 ring-rose-200 dark:bg-rose-950/40 dark:text-rose-200 dark:ring-rose-800',
    text: 'text-rose-800 dark:text-rose-200',
    border: 'ring-rose-200 dark:ring-rose-800',
    ring: 'ring-rose-400 dark:ring-rose-400',
    dot: 'bg-rose-500 dark:bg-rose-400',
    soft: 'bg-rose-50/60 dark:bg-rose-950/40',
    activeTab: 'bg-rose-600 text-white dark:bg-rose-700 dark:text-white',
  },
  curious: {
    short: 'Curious',
    title: 'Curious about',
    hint: 'Things you want to learn next',
    icon: 'i-heroicons-light-bulb',
    chip: 'bg-violet-50 text-violet-800 ring-violet-200 dark:bg-violet-950/40 dark:text-violet-200 dark:ring-violet-800',
    text: 'text-violet-800 dark:text-violet-200',
    border: 'ring-violet-200 dark:ring-violet-800',
    ring: 'ring-violet-400 dark:ring-violet-400',
    dot: 'bg-violet-500 dark:bg-violet-400',
    soft: 'bg-violet-50/60 dark:bg-violet-950/40',
    activeTab: 'bg-violet-600 text-white dark:bg-violet-700 dark:text-white',
  },
}
