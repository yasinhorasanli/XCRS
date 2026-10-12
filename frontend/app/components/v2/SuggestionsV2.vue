<script setup lang="ts">
import { CATEGORY_META } from '~/composables/categories'
import type { SkillGroup } from '~/types/apiV2'

const { chips, active, add, stopAttract } = useBoardV2()
const meta = computed(() => CATEGORY_META[active.value])
const api = useXcrsApiV2()

const { data } = await useAsyncData('skill-groups-v2', () => api.groups())
const groups = computed<SkillGroup[]>(() => data.value?.groups ?? [])
const tab = ref(0)
const current = computed(() => groups.value[tab.value])
const placed = (id: string) => chips.value.find((c) => c.skill === id)?.category
</script>

<template>
  <!-- The skill chips wear the selected box's colour: a tap adds the skill there. -->
  <aside class="rounded-2xl bg-default p-4 shadow-xs ring-1 ring-default" aria-label="Suggested skills">
    <h2 class="font-semibold">Quick start</h2>
    <p class="mt-0.5 text-xs text-muted">
      Starter skills for each field. Tap one to add it to
      <span class="font-medium" :class="CATEGORY_META[active].text">“{{ CATEGORY_META[active].title }}”</span>.
    </p>
    <div class="mt-3 flex flex-wrap gap-1">
      <button
        v-for="(g, i) in groups"
        :key="g.family"
        type="button"
        class="rounded-full px-2.5 py-1 text-xs transition"
        :class="i === tab ? 'bg-slate-900 dark:bg-slate-700 text-white' : 'bg-elevated text-toned hover:bg-accented'"
        @click="tab = i; stopAttract()"
      >
        {{ g.name }}
      </button>
    </div>
    <div v-if="current" class="mt-3 flex flex-wrap gap-1.5">
      <button
        v-for="s in current.skills"
        :key="s.id"
        type="button"
        class="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-sm shadow-xs ring-1 ring-inset transition-colors duration-300 hover:-translate-y-px"
        :class="[meta.soft, meta.border, placed(s.id) ? 'opacity-90' : '']"
        :title="placed(s.id) ? `Already in “${CATEGORY_META[placed(s.id)!].title}”; click to move it` : 'Click to add'"
        @click="add(s.name, active, s)"
      >
        <span v-if="placed(s.id)" class="h-1.5 w-1.5 rounded-full" :class="CATEGORY_META[placed(s.id)!].dot" />
        {{ s.name }}
      </button>
    </div>
  </aside>
</template>
