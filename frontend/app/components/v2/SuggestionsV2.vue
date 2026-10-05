<script setup lang="ts">
import { CATEGORY_META } from '~/composables/categories'
import type { SkillGroup } from '~/types/apiV2'

const { chips, active, add } = useBoardV2()
const api = useXcrsApiV2()

const { data } = await useAsyncData('skill-groups-v2', () => api.groups())
const groups = computed<SkillGroup[]>(() => data.value?.groups ?? [])
const tab = ref(0)
const current = computed(() => groups.value[tab.value])
const placed = (id: string) => chips.value.find((c) => c.skill === id)?.category
</script>

<template>
  <aside class="rounded-2xl bg-default p-4 shadow-xs ring-1 ring-default" aria-label="Suggested skills">
    <h2 class="font-semibold">Quick start</h2>
    <p class="mt-0.5 text-xs text-muted">
      Skills each career family starts with. Click to add to
      <span class="font-medium" :class="CATEGORY_META[active].text">“{{ CATEGORY_META[active].title }}”</span>.
    </p>
    <div class="mt-3 flex flex-wrap gap-1">
      <button
        v-for="(g, i) in groups"
        :key="g.family"
        type="button"
        class="rounded-full px-2.5 py-1 text-xs transition"
        :class="i === tab ? 'bg-slate-900 dark:bg-slate-700 text-white' : 'bg-elevated text-toned hover:bg-accented'"
        @click="tab = i"
      >
        {{ g.name }}
      </button>
    </div>
    <div v-if="current" class="mt-3 flex flex-wrap gap-1.5">
      <button
        v-for="s in current.skills"
        :key="s.id"
        type="button"
        class="inline-flex items-center gap-1.5 rounded-lg bg-default px-2.5 py-1 text-sm shadow-xs ring-1 ring-inset ring-default transition hover:-translate-y-px hover:ring-indigo-300 dark:hover:ring-indigo-800"
        :class="placed(s.id) ? 'opacity-90' : ''"
        :title="placed(s.id) ? `Already in “${CATEGORY_META[placed(s.id)!].title}”; click to move it` : 'Click to add'"
        @click="add(s.name, active, s)"
      >
        <span v-if="placed(s.id)" class="h-1.5 w-1.5 rounded-full" :class="CATEGORY_META[placed(s.id)!].dot" />
        {{ s.name }}
      </button>
    </div>
  </aside>
</template>
