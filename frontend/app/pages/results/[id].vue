<script setup lang="ts">
import { CATEGORIES, CATEGORY_META } from '~/composables/categories'
import { PROFICIENCY_NAMES } from '~/composables/useXcrsApiV2'
import { apiError } from '~/composables/useAccount'
import type { SkillToAdd } from '~/types/apiV2'

// One page instance per result: "Update my results" navigates from one result to another, and `id` is read once.
definePageMeta({ key: (route) => route.fullPath })

const route = useRoute()
const id = route.params.id as string
const api = useXcrsApiV2()
const { chips, add, fromResult, asInput, experience } = useBoardV2()
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
const pollingStopped = ref(false)
onMounted(() => {
  const started = Date.now()
  poller = setInterval(async () => {
    if (!pending.value || Date.now() - started > POLL_LIMIT_MS) {
      pollingStopped.value = true
      return clearInterval(poller)
    }
    try {
      data.value = await api.recommendation(id)
    } catch {
      // keep what we have; the next tick retries
    }
  }, POLL_MS)
})
onBeforeUnmount(() => clearInterval(poller))

/** "Download PDF" prints the page (the browser's "Save as PDF"), once the explanations are in: when none is pending,
 * or the page has stopped waiting for them, so the button never stays blocked. Before printing (the button or
 * Ctrl/Cmd+P) collapsed sections open, colours turn light and the title becomes the file name; afterwards all of it
 * goes back. Interactive parts are hidden by `print:hidden`. */
const pdfReady = computed(() => !!data.value && (!pending.value || pollingStopped.value))
const createdOn = computed(() => (data.value?.created_at ?? new Date().toISOString()).slice(0, 10))
// The page's own address, for the printed copy. Only the browser knows the public one (TLS ends at Tailscale Funnel,
// before Caddy), so the link is rendered in the browser only: rendering it on the server too made Vue warn that the
// server's empty link and the browser's didn't match (a hydration mismatch).
const resultUrl = computed(() => (import.meta.client ? window.location.href.split('?')[0] : ''))
let restorePrint: (() => void) | undefined
function beforePrint() {
  if (restorePrint) return // Chrome can announce one print twice (e.g. "Save as PDF" after a preview); keep the first state
  const opened = [...document.querySelectorAll<HTMLDetailsElement>('main details:not([open])')]
  opened.forEach((d) => (d.open = true))
  const root = document.documentElement
  const wasDark = root.classList.contains('dark')
  if (wasDark) root.classList.replace('dark', 'light')
  const title = document.title
  document.title = `XCRS results ${createdOn.value}`
  restorePrint = () => {
    opened.forEach((d) => (d.open = false))
    if (wasDark) root.classList.replace('light', 'dark')
    document.title = title
    restorePrint = undefined
  }
}
const afterPrint = () => restorePrint?.()
onMounted(() => {
  window.addEventListener('beforeprint', beforePrint)
  window.addEventListener('afterprint', afterPrint)
})
onBeforeUnmount(() => {
  window.removeEventListener('beforeprint', beforePrint)
  window.removeEventListener('afterprint', afterPrint)
})
const downloadPdf = () => window.print()

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

/** The answers per box, in board order; typed entries show what they were matched to. */
const answerGroups = computed(() =>
  CATEGORIES.map((category) => ({ category, items: data.value?.matched.filter((m) => m.category === category) ?? [] })).filter(
    (g) => g.items.length,
  ),
)
const unrecognised = computed(() => data.value?.matched.filter((m) => m.method !== 'picked' && !m.skills.length).length ?? 0)

/** Back to the board with these answers (also works when the link was shared). */
function edit() {
  if (data.value && !chips.value.length) fromResult(data.value)
  navigateTo('/')
}

/** Skills added from this page (gaps → Curious, basics → any box): they go on the board, and "Update my results"
 * runs it again. The board starts from these results when it is empty (a shared link, a reload). */
const added = ref<string[]>([])
const updating = ref(false)
function onAdd(items: SkillToAdd[]) {
  if (data.value && !chips.value.length) fromResult(data.value)
  // Every add can be undone for a few seconds ("add all" is one click): the board and the bar go back as they were.
  const board = JSON.parse(JSON.stringify(chips.value)) as typeof chips.value
  const before = [...added.value]
  const names = []
  for (const it of items) {
    const r = add(it.skill.name, it.category, it.skill, it.proficiency)
    if (r === 'added' || r === 'moved') names.push(it.skill.name)
    if (r === 'full') {
      toast.add({ title: 'Your board is full', description: 'Remove a skill on the board to add more.', color: 'warning' })
      break
    }
  }
  added.value = [...added.value, ...names.filter((n) => !added.value.includes(n))]
  if (!names.length) return
  const box = CATEGORY_META[items[0]!.category].title
  toast.add({
    title: names.length === 1 ? `Added “${names[0]}” to your board` : `Added ${names.length} skills to your board`,
    description: names.length === 1 ? `Under “${box}”.` : `Under “${box}”: ${names.slice(0, 3).join(', ')}${names.length > 3 ? '…' : ''}`,
    icon: 'i-heroicons-plus-circle',
    duration: 5000,
    actions: [
      {
        label: 'Undo',
        color: 'neutral',
        variant: 'outline',
        onClick: () => {
          chips.value = board
          added.value = before
        },
      },
    ],
  })
}
async function update() {
  updating.value = true
  try {
    const result = await api.recommend(asInput(), experience.value)
    added.value = []
    await navigateTo(`/results/${result.id}`)
  } catch {
    toast.add({ title: 'Could not update the results', description: 'Please try again in a moment.', color: 'error' })
  } finally {
    updating.value = false
  }
}
</script>

<template>
  <div class="mx-auto max-w-5xl px-4 pb-16 pt-8">
    <div class="flex flex-wrap items-center gap-3">
      <h1 class="text-2xl font-bold tracking-tight">Your roles</h1>
      <div class="ml-auto flex flex-wrap items-center gap-2 print:hidden">
        <UBadge v-if="data?.saved" color="primary" variant="soft" size="lg" icon="i-heroicons-bookmark-solid" label="Saved to your account" />
        <UButton v-else-if="accounts && data?.can_save" color="primary" variant="soft" :loading="saving" icon="i-heroicons-bookmark" label="Save to my account" @click="save" />
        <UTooltip v-if="data && !error" :text="pdfReady ? 'Save these results as a PDF' : 'Available when the explanations are ready'">
          <span>
            <!-- A disabled button gets no mouse events; without them the tooltip on its wrapper never opens. -->
            <UButton color="neutral" variant="soft" icon="i-heroicons-arrow-down-tray" label="Download PDF" class="disabled:pointer-events-none" :disabled="!pdfReady" @click="downloadPdf" />
          </span>
        </UTooltip>
        <UButton color="neutral" variant="soft" icon="i-heroicons-pencil-square" label="Edit my answers" @click="edit" />
      </div>
    </div>

    <p v-if="data" class="hidden text-xs text-muted print:block">
      XCRS · {{ createdOn }} · <ClientOnly><a :href="resultUrl" class="break-all">{{ resultUrl }}</a></ClientOnly>
    </p>

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
        <V2RoleResultV2
          v-for="(r, i) in data.roles"
          :key="r.id"
          :role="r"
          :rank="i + 1"
          :recommendation-id="data.id"
          :class="i > 0 ? 'print:break-before-page' : ''"
          @add="onAdd"
        />
      </div>

      <details class="mt-8 rounded-2xl bg-default p-4 text-sm shadow-xs ring-1 ring-default print:break-before-page">
        <summary class="cursor-pointer font-medium">
          Your answers: {{ data.matched.length }} {{ data.matched.length === 1 ? 'skill' : 'skills' }}
          <span class="font-normal text-muted">· what these results are based on</span>
        </summary>
        <p class="mt-2 text-xs text-muted">
          Skills you typed in your own words show the catalog skills we read them as<span v-if="unrecognised">; {{ unrecognised }} matched no skill and didn't count</span>.
        </p>
        <div class="mt-3 grid gap-3">
          <div v-for="g in answerGroups" :key="g.category">
            <p class="flex items-center gap-1.5 text-xs font-medium text-toned">
              <UIcon :name="CATEGORY_META[g.category].icon" class="h-3.5 w-3.5" />{{ CATEGORY_META[g.category].title }}
            </p>
            <ul class="mt-1 flex flex-wrap gap-1.5">
              <li v-for="(m, i) in g.items" :key="i" class="rounded-full px-2.5 py-0.5 text-xs ring-1 ring-inset" :class="CATEGORY_META[g.category].chip">
                {{ m.text }}
                <template v-if="m.method !== 'picked'">
                  <span v-if="m.skills.length" class="opacity-90">→ {{ m.skills.map((s) => s.name).join(', ') }}</span>
                  <span v-else class="text-amber-700 dark:text-amber-200">→ no skill recognised</span>
                </template>
                <span v-if="m.proficiency" class="ml-0.5 opacity-90">· {{ PROFICIENCY_NAMES[m.proficiency]?.toLowerCase() }}</span>
              </li>
            </ul>
          </div>
        </div>
        <p class="mt-3 text-xs text-dimmed">Engine {{ data.algorithm_version }} · catalog {{ data.catalog_version }}</p>
      </details>
    </template>

    <div v-if="added.length" class="sticky bottom-4 z-20 mt-6 print:hidden flex flex-wrap items-center gap-3 rounded-2xl bg-slate-900 dark:bg-slate-800 dark:ring-1 dark:ring-slate-700 px-4 py-3 text-sm text-white shadow-lg">
      <UIcon name="i-heroicons-plus-circle" class="h-5 w-5 shrink-0 text-indigo-300 dark:text-indigo-300" />
      <span class="min-w-0 flex-1">
        {{ added.length }} {{ added.length === 1 ? 'skill' : 'skills' }} added to your board:
        <span class="text-slate-300">{{ added.slice(0, 4).join(', ') }}<template v-if="added.length > 4"> and {{ added.length - 4 }} more</template></span>
      </span>
      <UButton color="neutral" variant="ghost" class="text-white hover:bg-white/10" label="Open the board" to="/" />
      <UButton :loading="updating" trailing-icon="i-heroicons-arrow-right-20-solid" label="Update my results" @click="update" />
    </div>
  </div>
</template>
