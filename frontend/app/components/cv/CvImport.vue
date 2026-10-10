<script setup lang="ts">
// Import skills from a CV or LinkedIn's "Save to PDF" (ADR-0045): upload or paste → a background job read by our
// own language model → review → add to the board (with Undo). Signed-in users only; nothing from the file is kept.
import { CATEGORY_META } from '~/composables/categories'
import type { CvImportResult, Experience } from '~/types/apiV2'
import type { ReviewRow } from './CvReview.vue'

const { enabled, loggedIn, signInPath } = useAccount()
const { chips, experience, add, onBoard } = useBoardV2()
const api = useXcrsApiV2()
const toast = useToast()

const open = ref(false)
const step = ref<'input' | 'waiting' | 'review'>('input')
const mode = ref<'pdf' | 'text'>('pdf')
const file = ref<File | null | undefined>(null)
const pasted = ref('')
const error = ref('')
const sending = ref(false)
const position = ref(0)
const elapsed = ref(0)
const result = ref<CvImportResult | null>(null)
const rows = ref<ReviewRow[]>([])
const useExperience = ref(true)
let poll: ReturnType<typeof setTimeout> | undefined
let tick: ReturnType<typeof setInterval> | undefined

const MAX_BYTES = 2 * 1024 * 1024
const POLL_MS = 3000

function errorMessage(e: unknown, fallback: string) {
  const detail = (e as { data?: { detail?: unknown } })?.data?.detail
  if (detail && typeof detail === 'object' && 'message' in detail) return String((detail as { message: unknown }).message)
  if (typeof detail === 'string') return detail
  return fallback
}

function stopWaiting() {
  clearTimeout(poll)
  clearInterval(tick)
}

function reset() {
  stopWaiting()
  step.value = 'input'
  error.value = ''
  file.value = null
  pasted.value = ''
  result.value = null
  rows.value = []
}

async function start() {
  error.value = ''
  const body = new FormData()
  if (mode.value === 'pdf') {
    if (!file.value) return void (error.value = 'Choose a PDF first.')
    if (file.value.size > MAX_BYTES) return void (error.value = 'That file is larger than 2 MB. Paste the text instead.')
    body.append('file', file.value)
  } else {
    if (pasted.value.trim().length < 40) return void (error.value = 'Paste a bit more of your CV.')
    body.append('text', pasted.value)
  }
  sending.value = true
  try {
    const job = await api.cvImport(body)
    position.value = job.position
    elapsed.value = 0
    step.value = 'waiting'
    tick = setInterval(() => elapsed.value++, 1000)
    poll = setTimeout(() => check(job.id), POLL_MS)
  } catch (e) {
    error.value = errorMessage(e, 'The server did not answer. Please try again in a moment.')
  } finally {
    sending.value = false
  }
}

async function check(id: string) {
  try {
    const job = await api.cvImportStatus(id)
    position.value = job.position
    if (job.status === 'done' && job.result) return done(job.result)
    if (job.status === 'failed') {
      stopWaiting()
      step.value = 'input'
      error.value = 'We could not read your CV this time. Please try again later, or paste the text.'
      return
    }
  } catch (e) {
    stopWaiting()
    step.value = 'input'
    error.value = errorMessage(e, 'We lost track of this import. Please try again.')
    return
  }
  poll = setTimeout(() => check(id), POLL_MS)
}

function done(r: CvImportResult) {
  stopWaiting()
  result.value = r
  rows.value = r.suggestions.map((s) => {
    const already = onBoard({ id: s.skill, name: s.name })
    return { ...s, add: !already, category: 'neutral' as const, level: s.proficiency ?? undefined, already }
  })
  useExperience.value = !!r.experience && experience.value !== r.experience
  step.value = 'review'
  if (!open.value) {
    toast.add({
      title: 'Your CV has been read',
      description: `${r.suggestions.length} skills to review.`,
      icon: 'i-heroicons-document-check',
      actions: [{ label: 'Review', color: 'primary', onClick: () => { open.value = true } }],
    })
  }
}

const chosen = computed(() => rows.value.filter((r) => r.add && !r.already))

function addToBoard() {
  // One Undo puts the board back as it was (the same pattern as adding from results).
  const board = JSON.parse(JSON.stringify(chips.value)) as typeof chips.value
  const before: Experience | null = experience.value
  let added = 0
  for (const r of chosen.value) {
    const outcome = add(r.name, r.category, { id: r.skill, name: r.name }, r.category === 'curious' ? undefined : r.level)
    if (outcome === 'added') added++
    if (outcome === 'full') {
      toast.add({ title: 'Your board is full', description: 'Remove a skill on the board to add more.', color: 'warning' })
      break
    }
  }
  if (useExperience.value && result.value?.experience) experience.value = result.value.experience
  const boxes = [...new Set(chosen.value.map((r) => CATEGORY_META[r.category].title))]
  open.value = false
  reset()
  toast.add({
    title: `Added ${added} ${added === 1 ? 'skill' : 'skills'} from your CV`,
    description: boxes.length === 1 ? `Under “${boxes[0]}”.` : undefined,
    icon: 'i-heroicons-plus-circle',
    duration: 5000,
    actions: [{ label: 'Undo', color: 'neutral', variant: 'outline', onClick: () => { chips.value = board; experience.value = before } }],
  })
}

function close() {
  open.value = false
  if (step.value === 'review') reset() // a running import keeps polling and reports when it's done
}

onBeforeUnmount(stopWaiting)
</script>

<template>
  <template v-if="enabled">
    <UButton color="neutral" variant="soft" icon="i-heroicons-document-arrow-up" label="Import from CV or LinkedIn PDF" @click="open = true" />
    <UModal
      v-model:open="open"
      :title="step === 'review' ? 'Skills found in your CV' : 'Import skills from your CV'"
      :description="step === 'review' ? 'Choose what goes on your board, in which box, and how well you know it.' : 'We suggest skills, levels and years in software; you review them before anything is added.'"
      :ui="{ content: 'sm:max-w-2xl' }"
    >
      <template #body>
        <div v-if="!loggedIn" class="space-y-3 text-sm">
          <p>Importing a CV needs an account: it runs our own language model for a minute or two, so we keep it to signed-in users for now.</p>
          <UButton :to="signInPath('/')" label="Sign in to import" icon="i-heroicons-arrow-right-on-rectangle" />
        </div>

        <div v-else-if="step === 'input'" class="space-y-4">
          <div class="flex gap-1 rounded-lg bg-elevated p-1 text-sm" role="tablist">
            <button
              v-for="m in (['pdf', 'text'] as const)"
              :key="m"
              type="button"
              role="tab"
              :aria-selected="mode === m"
              class="flex-1 rounded-md px-3 py-1.5 font-medium"
              :class="mode === m ? 'bg-default shadow-xs' : 'text-muted'"
              @click="mode = m; error = ''"
            >
              {{ m === 'pdf' ? 'Upload a PDF' : 'Paste the text' }}
            </button>
          </div>
          <template v-if="mode === 'pdf'">
            <UFileUpload
              v-model="file"
              accept="application/pdf,.pdf"
              icon="i-heroicons-document-arrow-up"
              label="Drop your CV or LinkedIn PDF here"
              description="PDF, up to 2 MB and 5 pages"
              layout="list"
              class="min-h-36 w-full"
            />
            <details class="text-xs text-muted">
              <summary class="cursor-pointer font-medium">How do I get my LinkedIn profile as a PDF?</summary>
              <p class="mt-1">On LinkedIn, open your profile, click <b>More</b> (or <b>Resources</b>) under your name, then <b>Save to PDF</b>.</p>
            </details>
          </template>
          <UTextarea v-else v-model="pasted" :rows="10" :maxlength="20000" placeholder="Paste the text of your CV (useful for scanned PDFs and Word files)" class="w-full" />
          <p v-if="error" class="text-sm text-error" role="alert">{{ error }}</p>
          <p class="text-xs text-muted">
            Your file is read on our own server by our own language model. Nothing from it is stored: only the skills you
            add to your board are kept. <NuxtLink to="/privacy" class="underline">Privacy</NuxtLink>
          </p>
        </div>

        <div v-else-if="step === 'waiting'" class="space-y-3 py-4 text-sm" aria-live="polite">
          <UProgress animation="carousel" />
          <p class="font-medium">Reading your CV… this usually takes one to three minutes.</p>
          <p class="text-muted">
            <template v-if="position > 1">{{ position - 1 }} {{ position === 2 ? 'import' : 'imports' }} ahead of yours. </template>
            {{ elapsed }} s so far. You can close this window: we'll tell you when it's ready.
          </p>
        </div>

        <CvReview v-else-if="result" v-model:rows="rows" v-model:use-experience="useExperience" :result="result" />
      </template>

      <template #footer>
        <div class="flex w-full items-center justify-end gap-2">
          <UButton color="neutral" variant="ghost" :label="step === 'review' ? 'Discard' : 'Close'" @click="close" />
          <UButton v-if="loggedIn && step === 'input'" :loading="sending" label="Find my skills" trailing-icon="i-heroicons-arrow-right-20-solid" @click="start" />
          <UButton
            v-if="step === 'review'"
            :disabled="!chosen.length && !useExperience"
            :label="chosen.length ? `Add ${chosen.length} ${chosen.length === 1 ? 'skill' : 'skills'} to my board` : 'Apply'"
            icon="i-heroicons-plus"
            @click="addToBoard"
          />
        </div>
      </template>
    </UModal>
  </template>
</template>
