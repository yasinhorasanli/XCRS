<script setup lang="ts">
// Test profiles for the decider's manual checks (xcrs/data/test_profiles.yaml, /api/v2/dev). Not for users:
// the API answers 404 unless XCRS_DEV_TOOLS is on, and Caddy hides /dev from public (Funnel) requests.
import { CATEGORY_META } from '~/composables/categories'
import { LEVEL_NAMES } from '~/composables/useXcrsApiV2'
import type { Category } from '~/types/apiV2'

useHead({ title: 'Test profiles · XCRS', meta: [{ name: 'robots', content: 'noindex' }] })

interface DevChip { category: Category; skill: string; name: string; proficiency: number | null }
interface DevProfile { id: string; title: string; background: string; expect: string[]; level: string; experience: string | null; chips: DevChip[] }
interface DevRole { role: string; name: string; score: number; level: string | null; target_level: string | null; coverage: number; interest: number }

const { data: profiles, error } = await useFetch<DevProfile[]>('/api/v2/dev/profiles')
const { data: previews, refresh } = await useFetch<{ id: string; roles: DevRole[] }[]>('/api/v2/dev/profiles/preview')
const previewOf = (id: string) => previews.value?.find((p) => p.id === id)?.roles ?? []
const levelName = (l: string | null) => (l ? LEVEL_NAMES[l] ?? l : 'none')
const LEVELS = ['entry', 'mid', 'senior', 'staff']

/** Role: is the expected main role first, in the top 3, or missing. Level: the engine's level for that role. */
function verdict(p: DevProfile) {
  const roles = previewOf(p.id)
  const rank = roles.findIndex((r) => r.role === p.expect[0])
  const level = rank >= 0 ? roles[rank]!.level : null
  const diff = level ? LEVELS.indexOf(level) - LEVELS.indexOf(p.level) : null
  return { rank, level, diff }
}

const board = useBoardV2()
const running = ref<string | null>(null)

async function openOnBoard(p: DevProfile) {
  board.clear()
  for (const c of p.chips) board.add(c.name, c.category, { id: c.skill, name: c.name }, c.proficiency ?? undefined)
  board.experience.value = (p.experience as typeof board.experience.value) ?? null
  await navigateTo('/')
}

async function run(p: DevProfile) {
  running.value = p.id
  try {
    const { id } = await $fetch<{ id: string }>(`/api/v2/dev/profiles/${p.id}/run`, { method: 'POST' })
    await navigateTo(`/results/${id}`)
  } finally {
    running.value = null
  }
}

const summary = computed(() => {
  const list = profiles.value ?? []
  const v = list.map(verdict)
  return {
    total: list.length,
    first: v.filter((x) => x.rank === 0).length,
    top3: v.filter((x) => x.rank >= 0 && x.rank < 3).length,
    level: v.filter((x) => x.diff === 0).length,
  }
})
</script>

<template>
  <div class="mx-auto max-w-6xl px-4 py-8">
    <div class="flex flex-wrap items-end justify-between gap-3">
      <div>
        <p class="text-xs font-medium uppercase tracking-wide text-amber-700 dark:text-amber-200">Dev tools · only on localhost and the tailnet</p>
        <h1 class="mt-1 text-2xl font-semibold tracking-tight">Test profiles</h1>
        <p class="mt-1 max-w-3xl text-sm text-muted">
          What a career adviser would expect, next to what the engine says (scored live, nothing stored).
          <b>Open on board</b> loads the profile to edit; <b>Run</b> makes a real result, marked as a test, with explanations.
        </p>
      </div>
      <UButton color="neutral" variant="outline" icon="i-heroicons-arrow-path" label="Re-score" @click="refresh()" />
    </div>

    <UAlert v-if="error" class="mt-6" color="warning" title="Dev tools are off on this server" description="Set XCRS_DEV_TOOLS=true for the API, or open this page on localhost or the tailnet." />

    <template v-else-if="profiles">
      <div class="mt-5 flex flex-wrap gap-2 text-sm">
        <span class="rounded-full bg-elevated px-3 py-1">{{ summary.total }} profiles</span>
        <span class="rounded-full bg-emerald-50 dark:bg-emerald-950/40 px-3 py-1 text-emerald-800 dark:text-emerald-200">expected role first: {{ summary.first }}</span>
        <span class="rounded-full bg-emerald-50 dark:bg-emerald-950/40 px-3 py-1 text-emerald-800 dark:text-emerald-200">in top 3: {{ summary.top3 }}</span>
        <span class="rounded-full bg-indigo-50 dark:bg-indigo-950/40 px-3 py-1 text-indigo-800 dark:text-indigo-200">expected level: {{ summary.level }}</span>
      </div>

      <div class="mt-5 grid gap-3">
        <article v-for="p in profiles" :key="p.id" class="rounded-2xl bg-default p-4 shadow-xs ring-1 ring-default">
          <div class="flex flex-wrap items-start justify-between gap-3">
            <div class="min-w-0">
              <h2 class="font-semibold">{{ p.title }}</h2>
              <p class="text-sm text-muted">{{ p.background }}</p>
            </div>
            <div class="flex shrink-0 gap-2">
              <UButton size="sm" color="neutral" variant="outline" label="Open on board" @click="openOnBoard(p)" />
              <UButton size="sm" label="Run" :loading="running === p.id" @click="run(p)" />
            </div>
          </div>

          <div class="mt-3 flex flex-wrap gap-1.5">
            <span v-for="c in p.chips" :key="c.skill" class="rounded-full px-2 py-0.5 text-xs ring-1 ring-inset" :class="CATEGORY_META[c.category].chip">
              {{ c.name }}<span v-if="c.proficiency" class="opacity-90"> · {{ c.proficiency }}</span>
            </span>
          </div>

          <div class="mt-3 grid gap-3 text-sm md:grid-cols-[14rem_1fr]">
            <div>
              <div class="text-xs uppercase tracking-wide text-dimmed">Expected</div>
              <div class="font-medium">{{ p.expect.join(', ') }}</div>
              <div class="text-toned">starts at {{ levelName(p.level) }}</div>
              <div class="text-xs text-muted">experience: {{ p.experience ?? 'not given' }}</div>
            </div>
            <div>
              <div class="text-xs uppercase tracking-wide text-dimmed">Engine (top 5)</div>
              <ol class="mt-0.5 grid gap-0.5">
                <li v-for="(r, i) in previewOf(p.id)" :key="r.role" class="flex flex-wrap items-baseline gap-x-2 tabular-nums" :class="p.expect.includes(r.role) ? 'font-medium text-highlighted' : 'text-muted'">
                  <span class="w-4 text-dimmed">{{ i + 1 }}</span>
                  <span>{{ r.name }}</span>
                  <span class="rounded px-1.5 text-xs" :class="r.role === p.expect[0] ? (r.level === p.level ? 'bg-emerald-100 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-200' : 'bg-amber-100 dark:bg-amber-950/40 text-amber-800 dark:text-amber-200') : 'bg-elevated'">
                    level: {{ levelName(r.level) }}
                  </span>
                  <span class="text-xs text-dimmed">score {{ r.score.toFixed(3) }} · interest {{ Math.round(r.interest * 100) }}% · has {{ Math.round(r.coverage * 100) }}%</span>
                </li>
              </ol>
              <p class="mt-1 text-xs" :class="verdict(p).rank === 0 ? 'text-emerald-700 dark:text-emerald-200' : 'text-amber-700 dark:text-amber-200'">
                <template v-if="verdict(p).rank === 0">Expected role first</template>
                <template v-else-if="verdict(p).rank > 0">Expected role at #{{ verdict(p).rank + 1 }}</template>
                <template v-else>Expected role not in the top 5</template>
                <template v-if="verdict(p).rank >= 0"> · level {{ verdict(p).diff === 0 ? 'as expected' : verdict(p).diff === null ? 'none (shown as "start at entry")' : verdict(p).diff! > 0 ? 'higher than expected' : 'lower than expected' }}</template>
              </p>
            </div>
          </div>
        </article>
      </div>
    </template>
  </div>
</template>
