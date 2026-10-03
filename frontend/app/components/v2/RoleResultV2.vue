<script setup lang="ts">
import { CATEGORY_META } from '~/composables/categories'
import { LEVEL_NAMES, PROFICIENCY_NAMES } from '~/composables/useXcrsApiV2'
import type { RoleResult } from '~/types/apiV2'

const props = defineProps<{ role: RoleResult; rank: number; recommendationId: string }>()
const api = useXcrsApiV2()

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
      <div v-for="s in stages" :key="s.stage" class="mt-2">
        <p class="text-xs text-slate-400">{{ s.stage }}</p>
        <ul class="mt-1 flex flex-wrap gap-1.5">
          <li v-for="g in s.gaps" :key="g.skills.map((x) => x.id).join('|')" class="rounded-lg bg-slate-50 px-2.5 py-1 text-sm ring-1 ring-inset ring-slate-200">
            {{ g.skills.map((x) => x.name).join(' or ') }}
            <span class="text-[11px] text-slate-500">· {{ PROFICIENCY_NAMES[g.need]?.toLowerCase() }}<template v-if="g.have"> (you: {{ PROFICIENCY_NAMES[g.have]?.toLowerCase() }})</template></span>
          </li>
        </ul>
      </div>
    </div>

    <details v-if="role.basics?.length" class="mt-3 text-sm">
      <summary class="cursor-pointer text-slate-500 hover:text-slate-700">
        Assumed at your level: {{ role.basics.length }} {{ role.basics.length === 1 ? 'basic' : 'basics' }} you didn't list. Worth a quick check
      </summary>
      <ul class="mt-2 flex flex-wrap gap-1.5">
        <li v-for="g in role.basics" :key="g.skills.map((x) => x.id).join('|')" class="rounded-lg px-2.5 py-1 text-xs text-slate-600 ring-1 ring-inset ring-slate-200">
          {{ g.skills.map((x) => x.name).join(' or ') }}
          <span class="text-slate-400">· {{ PROFICIENCY_NAMES[g.need]?.toLowerCase() }}</span>
        </li>
      </ul>
    </details>
  </article>
</template>
