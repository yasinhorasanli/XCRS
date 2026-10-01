// One color per recommended role (at most 3), used by its card, its connector lines and its course tags.
// Deliberately different hues from the four input categories (emerald, slate, rose, violet).
export const ROLE_COLORS = [
  {
    hex: '#4f46e5',
    badge: 'bg-indigo-600 text-white',
    tag: 'bg-indigo-50 text-indigo-700 ring-indigo-200',
    bar: 'bg-indigo-500',
    ring: 'ring-indigo-400',
    dot: 'bg-indigo-500',
    text: 'text-indigo-700',
  },
  {
    hex: '#0d9488',
    badge: 'bg-teal-600 text-white',
    tag: 'bg-teal-50 text-teal-700 ring-teal-200',
    bar: 'bg-teal-500',
    ring: 'ring-teal-400',
    dot: 'bg-teal-500',
    text: 'text-teal-700',
  },
  {
    hex: '#d97706',
    badge: 'bg-amber-500 text-white',
    tag: 'bg-amber-50 text-amber-800 ring-amber-200',
    bar: 'bg-amber-500',
    ring: 'ring-amber-400',
    dot: 'bg-amber-500',
    text: 'text-amber-700',
  },
] as const

export type RoleColor = (typeof ROLE_COLORS)[number]

export const roleColor = (index: number): RoleColor => ROLE_COLORS[index % ROLE_COLORS.length]!
