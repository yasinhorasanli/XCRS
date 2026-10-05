<script setup lang="ts">
import { CATEGORIES, CATEGORY_META } from '~/composables/categories'
import { LEVEL_NAMES, PROFICIENCY_NAMES } from '~/composables/useXcrsApiV2'
import type { Category, Gap, RoleResult, SkillRef, SkillToAdd } from '~/types/apiV2'

const props = defineProps<{ role: RoleResult; rank: number; recommendationId: string }>()
const emit = defineEmits<{ add: [items: SkillToAdd[]] }>()
const api = useXcrsApiV2()
const { onBoard } = useBoardV2()

/** Job titles to search for (ADR-0044): the learner's own first; each opens a LinkedIn Jobs search. */
const titles = computed(() => {
  const own = props.role.title_for_you
  return [...(own ? [own] : []), ...(props.role.job_titles ?? []).filter((t) => t !== own)]
})
const jobSearch = (title: string) => `https://www.linkedin.com/jobs/search/?keywords=${encodeURIComponent(title)}`

const gapOnBoard = (g: Gap) => g.skills.some((s) => onBoard(s))
const CURIOUS_PILL =
  'ml-1 inline-flex items-center gap-0.5 rounded-full bg-violet-50 px-2 py-0.5 text-xs font-medium text-violet-700 ring-1 ring-inset ring-violet-200 transition hover:bg-violet-100'


/** A basic the learner can confirm: into any box, at the level the role assumes (curious has no level). */
function basicMenu(g: Gap) {
  const into = (skill: SkillRef) =>
    CATEGORIES.map((category) => ({
      label: CATEGORY_META[category].short,
      icon: CATEGORY_META[category].icon,
      onSelect: () => emit('add', [{ skill, category, proficiency: category === 'curious' ? undefined : g.need }]),
    }))
  const heading = (name: string) => ({ type: 'label' as const, label: `Add ${name} to your board as` })
  return g.skills.map((s) => [heading(s.name), ...into(s)])
}

/** A skill to learn goes to "Curious"; with a choice ("Kotlin or Java"), the learner picks one. */
const gapMenu = (g: Gap) => [
  [
    { type: 'label' as const, label: 'Add to “Curious about”' },
    ...g.skills.map((skill) => ({ label: skill.name, onSelect: () => emit('add', [{ skill, category: 'curious' as Category }]) })),
  ],
]

function addAllBasics() {
  const items = (props.role.basics ?? [])
    .filter((g) => !gapOnBoard(g))
    .map((g) => ({ skill: g.skills[0]!, category: 'neutral' as Category, proficiency: g.need }))
  if (items.length) emit('add', items)
}

const levelName = (id: string, title?: string | null) => title ?? LEVEL_NAMES[id] ?? id
const pct = (x: number) => `${Math.round(Math.max(0, Math.min(1, x)) * 100)}%`
const stages = computed(() => {
  const out: { stage: string; gaps: RoleResult['gaps'] }[] = []
  for (const g of props.role.gaps) {
    const last = out.at(-1)
    if (last && last.stage === g.stage) last.gaps.push(g)
    else out.push({ stage: g.stage, gaps: [g] })
  }
  return out
})

const vote = ref<1 | -1 | null>(null)
async function send(rating: 1 | -1) {
  if (vote.value) return
  try {
    await api.feedback(props.recommendationId, { role: props.role.id, rating })
    vote.value = rating
  } catch {
    // feedback is optional
  }
}
</script>

<template>
  <article class="rounded-2xl bg-white p-5 shadow-xs ring-1 ring-slate-200">
    <header class="flex flex-wrap items-start gap-3">
      <span class="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-indigo-600 text-sm font-bold text-white">{{ rank }}</span>
      <div class="min-w-0 flex-1">
        <h2 class="text-lg font-semibold">{{ role.name }}</h2>
        <p class="text-sm text-slate-500">
          <template v-if="role.level">Estimated level: <span class="font-medium text-slate-700">{{ levelName(role.level.id, role.level.title) }}</span></template>
          <template v-else>Start at <span class="font-medium text-slate-700">{{ levelName(role.target_level.id, role.target_level.title) }}</span></template>
          <span class="text-slate-400"> · an estimate from your input</span>
        </p>
      </div>
      <div class="flex items-center gap-1" aria-label="Was this role a good suggestion?">
        <UButton size="xs" color="neutral" :variant="vote === 1 ? 'solid' : 'ghost'" icon="i-heroicons-hand-thumb-up" aria-label="Good suggestion" :disabled="!!vote" @click="send(1)" />
        <UButton size="xs" color="neutral" :variant="vote === -1 ? 'solid' : 'ghost'" icon="i-heroicons-hand-thumb-down" aria-label="Not for me" :disabled="!!vote" @click="send(-1)" />
      </div>
    </header>

    <div v-if="role.explanation_status === 'pending'" class="mt-3 flex items-center gap-2 text-sm text-slate-500" aria-live="polite">
      <UIcon name="i-heroicons-arrow-path" class="h-4 w-4 animate-spin" />
      Writing an explanation for you…
    </div>
    <div v-else-if="role.explanation" class="mt-3 space-y-2 text-[15px] leading-relaxed text-slate-700" aria-live="polite">
      <p>{{ role.explanation }}</p>
      <p v-if="role.next_step" class="rounded-lg bg-indigo-50/60 px-3 py-2 text-sm text-slate-700">
        <span class="font-medium text-indigo-700">First step:</span> {{ role.next_step }}
      </p>
      <p class="text-[11px] text-slate-400">Written by a local language model from the facts below; it can be imperfect.</p>
    </div>

    <div class="mt-4 grid gap-3 sm:grid-cols-2">
      <div>
        <div class="flex justify-between text-xs text-slate-500"><span>Interest</span><span class="tabular-nums">{{ pct(role.interest) }}</span></div>
        <div class="mt-1 h-2 rounded-full bg-slate-100"><div class="h-2 rounded-full bg-violet-500" :style="{ width: pct(role.interest) }" /></div>
        <p class="mt-1 text-[11px] text-slate-400">How much of what you enjoy or are curious about this role uses</p>
      </div>
      <div>
        <div class="flex justify-between text-xs text-slate-500"><span>What you already have</span><span class="tabular-nums">{{ pct(role.coverage) }}</span></div>
        <div class="mt-1 h-2 rounded-full bg-slate-100"><div class="h-2 rounded-full bg-emerald-500" :style="{ width: pct(role.coverage) }" /></div>
        <p class="mt-1 text-[11px] text-slate-400">Of the role's skills, across its levels</p>
      </div>
    </div>

    <ol class="mt-4 flex flex-wrap gap-2 text-xs" aria-label="Coverage per level">
      <li
        v-for="lv in role.levels"
        :key="lv.id"
        class="flex items-center gap-2 rounded-lg px-2.5 py-1 ring-1 ring-inset"
        :class="role.level?.id === lv.id ? 'bg-indigo-50 ring-indigo-300' : 'ring-slate-200'"
      >
        <span>{{ levelName(lv.id, lv.title) }}</span>
        <span class="tabular-nums text-slate-500">{{ pct(lv.coverage) }}</span>
      </li>
    </ol>

    <div v-if="role.because.length" class="mt-4">
      <h3 class="text-xs font-medium uppercase tracking-wide text-slate-500">Because of</h3>
      <div class="mt-1.5 flex flex-wrap gap-1.5">
        <span v-for="b in role.because" :key="b.id" class="rounded-full px-2.5 py-0.5 text-xs ring-1 ring-inset" :class="CATEGORY_META[b.category].chip">
          {{ b.name }} <span class="opacity-60">· {{ CATEGORY_META[b.category].short.toLowerCase() }}</span>
        </span>
      </div>
    </div>

    <div v-if="titles.length" class="mt-4">
      <h3 class="text-xs font-medium uppercase tracking-wide text-slate-500">Job titles to search for</h3>
      <ul class="mt-1.5 flex flex-wrap gap-1.5">
        <li v-for="(t, i) in titles" :key="t">
          <a
            :href="jobSearch(t)"
            target="_blank"
            rel="noopener"
            class="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs ring-1 ring-inset transition hover:ring-indigo-300"
            :class="i === 0 && role.title_for_you ? 'bg-indigo-50 font-medium text-indigo-800 ring-indigo-200' : 'text-slate-700 ring-slate-200'"
            :title="`Search LinkedIn Jobs for “${t}”`"
          >
            <UIcon v-if="i === 0 && role.title_for_you" name="i-heroicons-sparkles" class="h-3.5 w-3.5" />
            {{ t }}
            <UIcon name="i-heroicons-arrow-top-right-on-square" class="h-3 w-3 opacity-50" />
          </a>
        </li>
      </ul>
      <p v-if="role.title_for_you" class="mt-1 text-[11px] text-slate-400">The first one comes from your strongest skills for this role; each opens a LinkedIn Jobs search.</p>
    </div>

    <div v-if="role.resources?.length" class="mt-4">
      <h3 class="text-xs font-medium uppercase tracking-wide text-slate-500">Start learning</h3>
      <ul class="mt-1.5 grid gap-2 sm:grid-cols-3">
        <li v-for="res in role.resources" :key="res.id">
          <a
            :href="res.url"
            target="_blank"
            rel="noopener"
            class="flex h-full flex-col rounded-xl p-3 ring-1 ring-inset ring-slate-200 transition hover:-translate-y-px hover:ring-indigo-300"
          >
            <span class="flex items-center gap-1.5 text-[11px] text-slate-500">
              <span class="rounded bg-slate-100 px-1.5 py-0.5 capitalize">{{ res.type }}</span>
              <span v-if="res.free" class="rounded bg-emerald-50 px-1.5 py-0.5 text-emerald-700">Free</span>
              <span class="truncate">{{ res.provider }}</span>
            </span>
            <span class="mt-1 text-sm font-medium leading-snug">{{ res.title }}</span>
            <span class="mt-auto pt-1 text-[11px] text-slate-400">For {{ res.skills.map((s) => s.name).join(', ') }}</span>
          </a>
        </li>
      </ul>
    </div>

    <div v-if="role.gaps.length" class="mt-4">
      <h3 class="text-xs font-medium uppercase tracking-wide text-slate-500">
        <template v-if="role.level?.id === role.target_level.id">Still to cover at {{ levelName(role.target_level.id, role.target_level.title) }}</template>
        <template v-else>To reach {{ levelName(role.target_level.id, role.target_level.title) }}</template>: {{ role.gaps_total }} {{ role.gaps_total === 1 ? 'skill' : 'skills' }} to learn<span v-if="role.gaps_total > role.gaps.length">, first {{ role.gaps.length }}</span>
      </h3>
      <p class="mt-1 text-xs text-slate-500">
        Want to learn one? <span class="font-medium text-violet-700">+ Curious</span> puts it on your board under “Curious about”, then you can update your results.
      </p>
      <div v-for="s in stages" :key="s.stage" class="mt-2">
        <p class="text-xs text-slate-400">{{ s.stage }}</p>
        <ul class="mt-1 flex flex-wrap gap-1.5">
          <li v-for="g in s.gaps" :key="g.skills.map((x) => x.id).join('|')" class="flex items-center gap-1 rounded-lg bg-slate-50 py-1 pl-2.5 pr-1 text-sm ring-1 ring-inset ring-slate-200">
            <span>
              {{ g.skills.map((x) => x.name).join(' or ') }}
              <span class="text-[11px] text-slate-500">· {{ PROFICIENCY_NAMES[g.need]?.toLowerCase() }}<template v-if="g.have"> (you: {{ PROFICIENCY_NAMES[g.have]?.toLowerCase() }})</template></span>
            </span>
            <UIcon v-if="gapOnBoard(g)" name="i-heroicons-check-20-solid" class="mx-1 h-4 w-4 text-emerald-600" aria-label="On your board" />
            <button
              v-else-if="g.skills.length === 1"
              type="button"
              :class="CURIOUS_PILL"
              :aria-label="`Add ${g.skills[0]!.name} to Curious about`"
              @click="emit('add', [{ skill: g.skills[0]!, category: 'curious' }])"
            >
              <UIcon name="i-heroicons-plus-20-solid" class="h-3.5 w-3.5" />Curious
            </button>
            <UDropdownMenu v-else :items="gapMenu(g)">
              <button type="button" :class="CURIOUS_PILL" :aria-label="`Add one of ${g.skills.map((x) => x.name).join(', ')} to Curious about`">
                <UIcon name="i-heroicons-plus-20-solid" class="h-3.5 w-3.5" />Curious
              </button>
            </UDropdownMenu>
          </li>
        </ul>
      </div>
    </div>

    <details v-if="role.basics?.length" class="mt-3 text-sm">
      <summary class="cursor-pointer text-slate-500 hover:text-slate-700">
        Assumed at your level: {{ role.basics.length }} {{ role.basics.length === 1 ? 'basic' : 'basics' }} you didn't list. Worth a quick check
      </summary>
      <p class="mt-2 text-xs text-slate-500">
        Click one to put it on your board, or
        <button type="button" class="font-medium text-indigo-600 hover:underline" @click="addAllBasics">add all as Neutral</button>.
      </p>
      <ul class="mt-2 flex flex-wrap gap-1.5">
        <li v-for="g in role.basics" :key="g.skills.map((x) => x.id).join('|')">
          <span v-if="gapOnBoard(g)" class="inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs text-slate-400 ring-1 ring-inset ring-slate-200">
            <UIcon name="i-heroicons-check-20-solid" class="h-3.5 w-3.5 text-emerald-600" />
            {{ g.skills.map((x) => x.name).join(' or ') }}
          </span>
          <UDropdownMenu v-else :items="basicMenu(g)">
            <button type="button" class="inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs text-slate-600 ring-1 ring-inset ring-slate-200 transition hover:bg-slate-50 hover:ring-indigo-300">
              {{ g.skills.map((x) => x.name).join(' or ') }}
              <span class="text-slate-400">· {{ PROFICIENCY_NAMES[g.need]?.toLowerCase() }}</span>
              <UIcon name="i-heroicons-plus-20-solid" class="h-3.5 w-3.5 text-slate-400" />
            </button>
          </UDropdownMenu>
        </li>
      </ul>
    </details>
  </article>
</template>
