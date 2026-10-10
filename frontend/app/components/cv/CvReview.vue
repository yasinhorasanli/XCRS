<script setup lang="ts">
// The review step of a CV import (ADR-0045): every suggestion with a checkbox, a box (Neutral by default: a CV
// says what you know, not what you enjoyed) and level dots, grouped by job. Evidence is shown as plain text.
import { CATEGORIES, CATEGORY_META } from '~/composables/categories'
import { EXPERIENCE_OPTIONS, PROFICIENCY_NAMES } from '~/composables/useXcrsApiV2'
import type { Category, CvImportResult, CvSuggestion } from '~/types/apiV2'

export interface ReviewRow extends CvSuggestion {
  add: boolean
  category: Category
  level?: number
  already: boolean
}

const props = defineProps<{ result: CvImportResult }>()
const rows = defineModel<ReviewRow[]>('rows', { required: true })
const useExperience = defineModel<boolean>('useExperience', { required: true })

const boxItems = CATEGORIES.map((c) => ({ label: CATEGORY_META[c].short, value: c }))
const showIgnored = ref(false)

const period = (start: string | null, end: string | null) => {
  const year = (v: string | null) => (v ? v.slice(0, 4) : '')
  return start ? `${year(start)}–${end ? year(end) : 'now'}` : ''
}

const groups = computed(() => {
  const out: { key: string; title: string; subtitle: string; rows: ReviewRow[] }[] = []
  props.result.jobs.forEach((job, i) => {
    const jobRows = rows.value.filter((r) => r.job === i)
    if (jobRows.length) {
      out.push({ key: `job-${i}`, title: job.title, subtitle: [job.employer, period(job.start, job.end)].filter(Boolean).join(' · '), rows: jobRows })
    }
  })
  const other = rows.value.filter((r) => r.job === null || r.job >= props.result.jobs.length)
  if (other.length) out.push({ key: 'other', title: 'Elsewhere in your CV', subtitle: 'Summary, skills list or education', rows: other })
  return out
})

const experienceLabel = computed(() => EXPERIENCE_OPTIONS.find((o) => o.value === props.result.experience)?.label)
const selectable = computed(() => rows.value.filter((r) => !r.already))
const allChecked = computed(() => selectable.value.length > 0 && selectable.value.every((r) => r.add))
function toggleAll() {
  const value = !allChecked.value
  for (const r of selectable.value) r.add = value
}
function setLevel(row: ReviewRow, n: number) {
  row.level = row.level === n ? undefined : n
}
</script>

<template>
  <div class="space-y-4">
    <UAlert
      v-if="result.warnings.hidden || result.warnings.instructions"
      color="warning"
      variant="soft"
      icon="i-heroicons-shield-exclamation"
      title="We ignored part of this file"
    >
      <template #description>
        <p>
          <template v-if="result.warnings.hidden">{{ result.warnings.hidden }} {{ result.warnings.hidden === 1 ? 'piece' : 'pieces' }} of hidden text</template>
          <template v-if="result.warnings.hidden && result.warnings.instructions"> and </template>
          <template v-if="result.warnings.instructions">{{ result.warnings.instructions }} {{ result.warnings.instructions === 1 ? 'line' : 'lines' }} that looked like instructions to an AI</template>.
          Suggestions come only from the visible text.
        </p>
        <button type="button" class="mt-1 text-xs font-medium underline" @click="showIgnored = !showIgnored">
          {{ showIgnored ? 'Hide' : 'Show' }} what we ignored
        </button>
        <ul v-if="showIgnored" class="mt-1 list-disc space-y-0.5 pl-5 text-xs">
          <li v-for="(e, i) in result.warnings.examples" :key="i" class="break-words">{{ e }}</li>
        </ul>
      </template>
    </UAlert>

    <p v-if="!rows.length" class="text-sm text-muted">We found no skills from our catalog in this CV.</p>

    <label v-if="result.experience" class="flex items-center gap-2 rounded-lg bg-elevated px-3 py-2 text-sm">
      <UCheckbox v-model="useExperience" />
      <span>Set <b>years in software</b> to <b>{{ experienceLabel }}</b> <span class="text-muted">(from the dates of your tech jobs)</span></span>
    </label>

    <div v-if="rows.length" class="flex items-center justify-between text-xs text-muted">
      <span>Pick a box and a level for each skill. Levels are suggested from years of use.</span>
      <button type="button" class="font-medium text-primary hover:underline" @click="toggleAll">{{ allChecked ? 'Select none' : 'Select all' }}</button>
    </div>

    <section v-for="g in groups" :key="g.key" class="rounded-xl border border-default">
      <header class="border-b border-default px-3 py-2">
        <p class="text-sm font-semibold text-highlighted">{{ g.title }}</p>
        <p v-if="g.subtitle" class="text-xs text-muted">{{ g.subtitle }}</p>
      </header>
      <ul class="divide-y divide-default">
        <li v-for="row in g.rows" :key="row.skill" class="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2" :class="row.already ? 'opacity-60' : ''">
          <UCheckbox v-model="row.add" :disabled="row.already" :aria-label="`Add ${row.name}`" />
          <div class="min-w-0 flex-1">
            <p class="text-sm font-medium">
              {{ row.name }}
              <span v-if="row.years" class="ml-1 text-xs font-normal text-muted">{{ row.years }} {{ row.years === 1 ? 'yr' : 'yrs' }}</span>
              <UBadge v-if="row.already" size="sm" color="neutral" variant="soft" class="ml-1">On your board</UBadge>
            </p>
            <p class="truncate text-xs text-muted" :title="row.evidence">“{{ row.evidence }}”</p>
          </div>
          <template v-if="!row.already">
            <USelect v-model="row.category" :items="boxItems" size="xs" class="w-28" :aria-label="`Box for ${row.name}`" />
            <div class="flex w-16 items-center gap-1" :class="row.category === 'curious' ? 'invisible' : ''" role="group" :aria-label="`How well you know ${row.name}`">
              <button
                v-for="n in 4"
                :key="n"
                type="button"
                class="size-3 rounded-full ring-1 ring-inset ring-current text-primary"
                :class="(row.level ?? 0) >= n ? 'bg-current' : 'opacity-40'"
                :title="PROFICIENCY_NAMES[n]"
                :aria-pressed="row.level === n"
                @click="setLevel(row, n)"
              />
            </div>
          </template>
        </li>
      </ul>
    </section>
  </div>
</template>
