<script setup lang="ts">
import { CATEGORIES, CATEGORY_META } from '~/composables/categories'
import { EXPERIENCE_OPTIONS, LEVEL_NAMES, PROFICIENCY_NAMES, PROVIDER_META } from '~/composables/useXcrsApiV2'
import { apiError } from '~/composables/useAccount'
import type { Me, ResultSummary, SavedBoard } from '~/types/apiV2'

const api = useXcrsApiV2()
const toast = useToast()
const { loggedIn, clear: forgetSession, signInPath, signOut } = useAccount()

const me = ref<Me | null>(null)
const results = ref<ResultSummary[]>([])
const board = ref<SavedBoard | null>(null)
/** The saved board's skills per box, in the board's order (ADR-0043). */
const skillGroups = computed(() =>
  CATEGORIES.map((category) => ({ category, chips: board.value?.chips.filter((c) => c.category === category) ?? [] })).filter(
    (g) => g.chips.length,
  ),
)
const experienceLabel = computed(() => EXPERIENCE_OPTIONS.find((o) => o.value === board.value?.experience)?.label)
const loading = ref(true)
const confirmDelete = ref(false)
const busy = ref(false)

useHead({ title: 'Your account · XCRS' })

async function ended() {
  await forgetSession()
  await navigateTo(signInPath('/account'))
}

async function load() {
  if (!loggedIn.value) return navigateTo(signInPath('/account'))
  try {
    let saved
    ;[me.value, results.value, saved] = await Promise.all([api.me(), api.results(), api.board()])
    board.value = saved.board
  } catch (e) {
    if (apiError(e).status === 401) return ended()
    toast.add({ title: 'Could not load your account', color: 'error', icon: 'i-heroicons-exclamation-triangle' })
  } finally {
    loading.value = false
  }
}
onMounted(load)

const when = (iso: string) => new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
const roleLine = (r: ResultSummary) =>
  r.roles.length
    ? r.roles.map((role) => (role.level ? `${role.name} (${LEVEL_NAMES[role.level] ?? role.level})` : role.name)).join(' · ')
    : 'No skills recognised'

async function removeResult(id: string) {
  try {
    await api.deleteResult(id)
    results.value = results.value.filter((r) => r.id !== id)
    toast.add({ title: 'Result deleted', icon: 'i-heroicons-trash' })
  } catch {
    toast.add({ title: 'Could not delete the result', color: 'error' })
  }
}

async function signOutEverywhere() {
  busy.value = true
  try {
    await api.signOutEverywhere()
    await forgetSession()
    toast.add({ title: 'Signed out on all devices', icon: 'i-heroicons-check' })
    await navigateTo('/')
  } catch {
    toast.add({ title: 'Could not sign out everywhere', color: 'error' })
  } finally {
    busy.value = false
  }
}

async function deleteAccount() {
  busy.value = true
  try {
    await api.deleteAccount()
    await forgetSession()
    confirmDelete.value = false
    toast.add({ title: 'Your account and its data were deleted', icon: 'i-heroicons-check' })
    await navigateTo('/')
  } catch {
    toast.add({ title: 'Could not delete the account', color: 'error' })
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="mx-auto max-w-3xl px-4 pb-16 pt-8">
    <h1 class="text-2xl font-bold tracking-tight">Your account</h1>

    <div v-if="loading" class="mt-6 grid gap-3">
      <USkeleton class="h-24 w-full" />
      <USkeleton class="h-40 w-full" />
    </div>

    <template v-else-if="me">
      <section class="mt-6 rounded-2xl bg-white p-5 shadow-xs ring-1 ring-slate-200">
        <p class="text-lg font-semibold">{{ me.display_name || 'No name given' }}</p>
        <p class="text-sm text-slate-600">{{ me.email || 'No verified email' }}</p>
        <div class="mt-3 flex flex-wrap items-center gap-2 text-sm text-slate-500">
          Signed in with
          <UBadge v-for="p in me.providers" :key="p" color="neutral" variant="soft" :icon="PROVIDER_META[p].icon" :label="PROVIDER_META[p].label" />
          <span>· member since {{ new Date(me.created_at).toLocaleDateString() }}</span>
        </div>
      </section>

      <section class="mt-6">
        <div class="flex items-center gap-3">
          <h2 class="text-lg font-semibold">Your skills</h2>
          <UButton class="ml-auto" to="/" size="sm" color="neutral" variant="soft" icon="i-heroicons-pencil-square" label="Edit on the board" />
        </div>
        <p v-if="experienceLabel" class="mt-1 text-sm text-slate-500">Years in software: {{ experienceLabel }}</p>
        <p v-if="!skillGroups.length" class="mt-3 text-sm text-slate-500">
          No saved board yet. The board you use while signed in is kept here.
        </p>
        <div v-else class="mt-3 grid gap-3">
          <div v-for="g in skillGroups" :key="g.category" class="rounded-2xl p-4 ring-1 ring-inset" :class="[CATEGORY_META[g.category].soft, CATEGORY_META[g.category].chip.split(' ').find((c) => c.startsWith('ring-'))]">
            <h3 class="flex items-center gap-2 text-sm font-semibold">
              <UIcon :name="CATEGORY_META[g.category].icon" class="h-4 w-4" />
              {{ CATEGORY_META[g.category].title }}
              <span class="font-normal text-slate-500">{{ g.chips.length }}</span>
            </h3>
            <ul class="mt-2 flex flex-wrap gap-1.5">
              <li v-for="c in g.chips" :key="`${c.skill ?? ''}|${c.text ?? ''}`" class="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-sm ring-1 ring-inset" :class="CATEGORY_META[g.category].chip">
                {{ c.name ?? c.text ?? c.skill }}
                <span v-if="c.proficiency" class="flex items-center gap-0.5" :title="PROFICIENCY_NAMES[c.proficiency]" :aria-label="PROFICIENCY_NAMES[c.proficiency]">
                  <span v-for="n in 4" :key="n" class="h-2 w-2 rounded-full ring-1 ring-current" :class="c.proficiency >= n ? 'bg-current opacity-80' : 'opacity-40'" />
                </span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      <section class="mt-8">
        <div class="flex items-center gap-3">
          <h2 class="text-lg font-semibold">Your results</h2>
        </div>
        <p v-if="!results.length" class="mt-3 text-sm text-slate-500">
          No saved results yet. Results you make while signed in are kept here.
        </p>
        <ul v-else class="mt-3 grid gap-2">
          <li v-for="r in results" :key="r.id" class="flex items-center gap-3 rounded-xl bg-white px-4 py-3 shadow-xs ring-1 ring-slate-200">
            <NuxtLink :to="`/results/${r.id}`" class="min-w-0 flex-1">
              <p class="truncate font-medium text-slate-800 hover:text-indigo-600">{{ roleLine(r) }}</p>
              <p class="text-xs text-slate-500">{{ when(r.created_at) }}</p>
            </NuxtLink>
            <UButton color="neutral" variant="ghost" size="sm" icon="i-heroicons-trash" aria-label="Delete this result" @click="removeResult(r.id)" />
          </li>
        </ul>
      </section>

      <section class="mt-8 rounded-2xl bg-white p-5 shadow-xs ring-1 ring-slate-200">
        <h2 class="text-lg font-semibold">Your data</h2>
        <p class="mt-1 text-sm text-slate-600">
          Download everything we keep about you, or delete it. Read the <NuxtLink to="/privacy" class="text-indigo-600 underline">privacy notice</NuxtLink>.
        </p>
        <div class="mt-4 flex flex-wrap gap-2">
          <UButton to="/api/v2/me/export" external download color="neutral" variant="outline" icon="i-heroicons-arrow-down-tray" label="Download my data (JSON)" />
          <UButton color="neutral" variant="outline" icon="i-heroicons-arrow-left-start-on-rectangle" label="Sign out" @click="signOut()" />
          <UButton color="neutral" variant="outline" :loading="busy" icon="i-heroicons-device-phone-mobile" label="Sign out on all devices" @click="signOutEverywhere" />
          <UButton color="error" variant="soft" icon="i-heroicons-trash" label="Delete my account" @click="confirmDelete = true" />
        </div>
      </section>
    </template>

    <UModal v-model:open="confirmDelete" title="Delete your account?">
      <template #body>
        <p class="text-sm text-slate-600">
          This deletes your account, your sign-in accounts' link to XCRS, your board, your {{ results.length }}
          {{ results.length === 1 ? 'result' : 'results' }} and the feedback on them. It can't be undone.
        </p>
      </template>
      <template #footer>
        <div class="flex w-full justify-end gap-2">
          <UButton color="neutral" variant="ghost" label="Keep my account" @click="confirmDelete = false" />
          <UButton color="error" :loading="busy" icon="i-heroicons-trash" label="Delete everything" @click="deleteAccount" />
        </div>
      </template>
    </UModal>
  </div>
</template>
