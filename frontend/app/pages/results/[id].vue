<script setup lang="ts">
import { CATEGORY_META } from '~/composables/categories'
import { apiError } from '~/composables/useAccount'

const route = useRoute()
const id = route.params.id as string
const api = useXcrsApiV2()
const { chips, add, clear, experience } = useBoardV2()
const { enabled: accounts, loggedIn, signInPath } = useAccount()
const toast = useToast()

const { data, error } = await useAsyncData(`recommendation-v2-${id}`, () => api.recommendation(id))
if (error.value && import.meta.server) {
  const event = useRequestEvent()
  if (event) setResponseStatus(event, 404)
}
// Explanations are written in the background (ADR-0037): poll while any role's is still pending.
const pending = computed(() => data.value?.roles.filter((r) => r.explanation_status === 'pending').length ?? 0)
const POLL_MS = 2500
const POLL_LIMIT_MS = 6 * 60 * 1000
let poller: ReturnType<typeof setInterval> | undefined
onMounted(() => {
  const started = Date.now()
  poller = setInterval(async () => {
    if (!pending.value || Date.now() - started > POLL_LIMIT_MS) return clearInterval(poller)
    try {
      data.value = await api.recommendation(id)
    } catch {
      // keep what we have; the next tick retries
    }
  }, POLL_MS)
})
onBeforeUnmount(() => clearInterval(poller))

/** Keep an anonymous result in the account (ADR-0043); signed out, sign in first and come back to save it. */
const saving = ref(false)
async function save() {
  if (!loggedIn.value) return navigateTo(signInPath(`/results/${id}?save=1`))
  saving.value = true
  try {
    await api.saveResult(id)
    data.value = await api.recommendation(id)
    toast.add({ title: 'Saved to your account', icon: 'i-heroicons-bookmark', actions: [{ label: 'My account', to: '/account', color: 'neutral', variant: 'outline' }] })
  } catch (e) {
    toast.add({ title: 'Could not save these results', description: apiError(e).message, color: 'error', icon: 'i-heroicons-exclamation-triangle' })
  } finally {
    saving.value = false
  }
}
onMounted(() => {
  if (route.query.save && loggedIn.value && data.value?.can_save) save()
})

useHead(() => ({ title: data.value?.roles[0] ? `${data.value.roles[0].name} and more · XCRS` : 'Your results · XCRS' }))

/** Back to the board with these answers (also works when the link was shared). */
function edit() {
  if (data.value && !chips.value.length) {
    clear()
    for (const m of data.value.matched) {
      const picked = m.method === 'picked' ? m.skills[0] : undefined
      add(m.text ?? picked?.name ?? '', m.category, picked, m.proficiency ?? undefined)
    }
    experience.value = data.value.experience ?? null
  }
  navigateTo('/')
}
</script>

<template>
  <div class="mx-auto max-w-5xl px-4 pb-16 pt-8">
    <div class="flex flex-wrap items-center gap-3">
      <h1 class="text-2xl font-bold tracking-tight">Your roles</h1>
      <div class="ml-auto flex flex-wrap items-center gap-2">
        <UBadge v-if="data?.saved" color="primary" variant="soft" size="lg" icon="i-heroicons-bookmark-solid" label="Saved to your account" />
        <UButton v-else-if="accounts && data?.can_save" color="primary" variant="soft" :loading="saving" icon="i-heroicons-bookmark" label="Save to my account" @click="save" />
        <UButton color="neutral" variant="soft" icon="i-heroicons-pencil-square" label="Edit my answers" @click="edit" />
      </div>
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
