<script setup lang="ts">
import type { KnowledgeUnit, KnowledgeUnitGroup, RelatedUnit } from '~/types/api'

const { all, active, add, categoryOf } = useBoard()
const api = useXcrsApi()

const tab = ref<'for-you' | 'browse'>('browse')

// --- Browse: curated groups (loaded once) ---
const { data: groupData } = await useAsyncData('knowledge-unit-groups', () => api.groups(), {
  default: () => ({ groups: [] as KnowledgeUnitGroup[] }),
})
const openGroup = ref<string | null>(null)
const groups = computed(() => groupData.value.groups)

// --- For you: related to what's on the board, refreshed as it changes ---
const related = ref<RelatedUnit[]>([])
const relatedLoading = ref(false)
let relatedTimer: ReturnType<typeof setTimeout> | undefined
let relatedCall = 0

watch(
  () => all.value.join('|'),
  () => {
    clearTimeout(relatedTimer)
    if (!all.value.length) {
      related.value = []
      return
    }
    relatedTimer = setTimeout(async () => {
      const call = ++relatedCall
      relatedLoading.value = true
      try {
        const { units } = await api.related(all.value, 16)
        if (call === relatedCall) related.value = units
      } catch {
        // suggestions are optional; keep the previous ones
      } finally {
        if (call === relatedCall) relatedLoading.value = false
      }
    }, 500)
  },
  { immediate: true },
)
watch(
  () => all.value.length > 0,
  (hasSkills, hadSkills) => {
    if (hasSkills && !hadSkills) tab.value = 'for-you'
  },
)

// --- Search across curated labels and every roadmap concept ---
const query = ref('')
const results = ref<KnowledgeUnit[]>([])
let searchTimer: ReturnType<typeof setTimeout> | undefined
watch(query, (q) => {
  clearTimeout(searchTimer)
  if (q.trim().length < 2) {
    results.value = []
    return
  }
  searchTimer = setTimeout(async () => {
    try {
      results.value = (await api.search(q.trim(), 24)).units
    } catch {
      results.value = []
    }
  }, 180)
})
</script>

<template>
  <aside class="flex flex-col rounded-2xl bg-white shadow-sm ring-1 ring-slate-200">
    <div class="border-b border-slate-100 p-4">
      <h2 class="font-semibold">Suggestions</h2>
      <p class="mt-0.5 text-xs text-slate-500">Drag a skill into a box, or click it to add it to:</p>
      <div class="mt-2 grid grid-cols-4 gap-1 rounded-xl bg-slate-100 p-1" role="radiogroup" aria-label="Clicking a suggestion adds it to">
        <button
          v-for="c in CATEGORIES"
          :key="c"
          type="button"
          role="radio"
          :aria-checked="active === c"
          :aria-label="CATEGORY_META[c].title"
          class="truncate rounded-lg px-1.5 py-1 text-xs font-medium transition"
          :class="active === c ? CATEGORY_META[c].activeTab : 'text-slate-600 hover:bg-white'"
          @click="active = c"
        >
          {{ CATEGORY_META[c].short }}
        </button>
      </div>
      <UInput
        v-model="query"
        class="mt-3"
        icon="i-heroicons-magnifying-glass-20-solid"
        placeholder="Search skills, tools, concepts…"
        aria-label="Search suggestions"
        :ui="{ icon: { trailing: { pointer: '' } } }"
      >
        <template #trailing>
          <UButton v-show="query" color="gray" variant="link" icon="i-heroicons-x-mark-20-solid" :padded="false" aria-label="Clear search" @click="query = ''" />
        </template>
      </UInput>
    </div>

    <!-- Search results -->
    <div v-if="query.trim().length >= 2" class="max-h-[32rem] overflow-y-auto p-4">
      <p v-if="!results.length" class="text-sm text-slate-400">No match yet. Press Enter in a box to add “{{ query }}” as your own words.</p>
      <div class="flex flex-wrap gap-1.5">
        <BoardSkillChip
          v-for="u in results"
          :key="u.label"
          :label="u.label"
          :placed="categoryOf(u.label)"
          :hint="u.roles.length ? (u.roles.length === 1 ? u.roles[0] : `${u.roles.length} roadmaps`) : undefined"
          @pick="add(u.label)"
        />
      </div>
    </div>

    <template v-else>
      <div class="flex gap-4 border-b border-slate-100 px-4" role="tablist">
        <button
          v-for="t in [{ id: 'for-you', label: 'For you' }, { id: 'browse', label: 'Browse' }] as const"
          :key="t.id"
          type="button"
          role="tab"
          :aria-selected="tab === t.id"
          class="-mb-px border-b-2 py-2 text-sm font-medium transition"
          :class="tab === t.id ? 'border-indigo-600 text-indigo-700' : 'border-transparent text-slate-500 hover:text-slate-700'"
          @click="tab = t.id"
        >
          {{ t.label }}
          <span v-if="t.id === 'for-you' && related.length" class="ml-1 rounded-full bg-indigo-100 px-1.5 text-[10px] text-indigo-700">
            {{ related.length }}
          </span>
        </button>
      </div>

      <div class="max-h-[32rem] overflow-y-auto p-4">
        <div v-if="tab === 'for-you'">
          <p v-if="!all.length" class="text-sm text-slate-500">
            Add a few things you know, and this list fills with related skills from the career roadmaps.
          </p>
          <div v-else-if="relatedLoading && !related.length" class="flex flex-wrap gap-1.5">
            <span v-for="i in 8" :key="i" class="shimmer h-10 rounded-lg" :style="{ width: `${70 + ((i * 37) % 60)}px` }" />
          </div>
          <template v-else>
            <p class="mb-2 text-xs text-slate-500">Close to what you've added. The small text shows which of your skills each one relates to.</p>
            <div class="flex flex-wrap gap-1.5" :class="relatedLoading ? 'opacity-60' : ''">
              <BoardSkillChip
                v-for="u in related"
                :key="u.label"
                :label="u.label"
                :placed="categoryOf(u.label)"
                :hint="`near ${u.because}`"
                @pick="add(u.label)"
              />
            </div>
            <p v-if="!related.length" class="text-sm text-slate-400">Nothing close enough yet. Try adding another skill.</p>
          </template>
        </div>

        <div v-else class="space-y-1">
          <div v-for="g in groups" :key="g.name" class="rounded-xl ring-1 ring-slate-100">
            <button
              type="button"
              class="flex w-full items-center justify-between px-3 py-2 text-left text-sm font-medium hover:bg-slate-50"
              :aria-expanded="openGroup === g.name"
              @click="openGroup = openGroup === g.name ? null : g.name"
            >
              {{ g.name }}
              <UIcon name="i-heroicons-chevron-down-20-solid" class="h-4 w-4 text-slate-400 transition" :class="openGroup === g.name ? 'rotate-180' : ''" />
            </button>
            <div v-if="openGroup === g.name" class="flex flex-wrap gap-1.5 px-3 pb-3">
              <BoardSkillChip v-for="u in g.units" :key="u" :label="u" :placed="categoryOf(u)" @pick="add(u)" />
            </div>
          </div>
        </div>
      </div>
    </template>
  </aside>
</template>
