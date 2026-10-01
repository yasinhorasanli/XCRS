<script setup lang="ts">
import { CATEGORY_META } from '~/composables/useBoard'

const route = useRoute()
const id = route.params.id as string
const api = useXcrsApiV2()
const { chips, add, clear } = useBoardV2()

const { data, error } = await useAsyncData(`recommendation-v2-${id}`, () => api.recommendation(id))
if (error.value && import.meta.server) {
  const event = useRequestEvent()
  if (event) setResponseStatus(event, 404)
}
useHead(() => ({ title: data.value?.roles[0] ? `${data.value.roles[0].name} and more · XCRS` : 'Your results · XCRS' }))

/** Back to the board with these answers (also works when the link was shared). */
function edit() {
  if (data.value && !chips.value.length) {
    clear()
    for (const m of data.value.matched) {
      const picked = m.method === 'picked' ? m.skills[0] : undefined
      add(m.text ?? picked?.name ?? '', m.category, picked, m.proficiency ?? undefined)
    }
  }
  navigateTo('/v2')
}
</script>

<template>
  <div class="mx-auto max-w-5xl px-4 pb-16 pt-8">
    <div class="flex flex-wrap items-center gap-3">
      <h1 class="text-2xl font-bold tracking-tight">Your roles</h1>
      <span class="rounded-full bg-indigo-50 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-indigo-700 ring-1 ring-indigo-200">New engine · beta</span>
      <UButton class="ml-auto" color="neutral" variant="soft" icon="i-heroicons-pencil-square" label="Edit my answers" @click="edit" />
    </div>

    <UAlert v-if="error" class="mt-6" color="error" title="We couldn't find these results" description="The link may be wrong or the results were removed." />

    <template v-else-if="data">
      <UAlert
        v-if="data.status === 'insufficient_input'"
        class="mt-6"
        color="warning"
        title="We couldn't recognise any skills"
        description="Try naming tools, languages or topics (for example “SQL”, “React”, “testing”), or pick them from the suggestions."
      />
      <div class="mt-6 grid gap-5">
        <V2RoleResultV2 v-for="(r, i) in data.roles" :key="r.id" :role="r" :rank="i + 1" :recommendation-id="data.id" />
      </div>

      <details class="mt-8 rounded-2xl bg-white p-4 text-sm shadow-xs ring-1 ring-slate-200">
        <summary class="cursor-pointer font-medium">How we read your input</summary>
        <ul class="mt-3 grid gap-1.5">
          <li v-for="(m, i) in data.matched" :key="i" class="flex flex-wrap items-center gap-2">
            <span class="rounded-full px-2 py-0.5 text-xs ring-1 ring-inset" :class="CATEGORY_META[m.category].chip">{{ m.text }}</span>
            <span class="text-slate-400">→</span>
            <span v-if="m.skills.length">{{ m.skills.map((s) => s.name).join(', ') }}</span>
            <span v-else class="text-amber-700">no skill recognised</span>
          </li>
        </ul>
        <p class="mt-3 text-xs text-slate-400">Engine {{ data.algorithm_version }} · catalog {{ data.catalog_version }}</p>
      </details>
    </template>
  </div>
</template>
