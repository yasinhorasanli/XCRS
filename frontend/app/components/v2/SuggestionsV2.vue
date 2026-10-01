<script setup lang="ts">
import { CATEGORY_META } from '~/composables/useBoard'
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
  <aside class="rounded-2xl bg-white p-4 shadow-xs ring-1 ring-slate-200" aria-label="Suggested skills">
    <h2 class="font-semibold">Quick start</h2>
    <p class="mt-0.5 text-xs text-slate-500">
      Skills each career family starts with. Click to add to
      <span class="font-medium" :class="CATEGORY_META[active].chip.split(' ')[1]">“{{ CATEGORY_META[active].title }}”</span>.
    </p>
    <div class="mt-3 flex flex-wrap gap-1">
      <button
        v-for="(g, i) in groups"
        :key="g.family"
        type="button"
        class="rounded-full px-2.5 py-1 text-xs transition"
        :class="i === tab ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'"
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
        class="inline-flex items-center gap-1.5 rounded-lg bg-white px-2.5 py-1 text-sm shadow-xs ring-1 ring-inset ring-slate-200 transition hover:-translate-y-px hover:ring-indigo-300"
        :class="placed(s.id) ? 'opacity-60' : ''"
        :title="placed(s.id) ? `Already in “${CATEGORY_META[placed(s.id)!].title}”; click to move it` : 'Click to add'"
        @click="add(s.name, active, s)"
      >
        <span v-if="placed(s.id)" class="h-1.5 w-1.5 rounded-full" :class="CATEGORY_META[placed(s.id)!].dot" />
        {{ s.name }}
      </button>
    </div>
  </aside>
</template>
