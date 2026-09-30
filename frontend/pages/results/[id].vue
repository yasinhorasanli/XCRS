<script setup lang="ts">
import type { Category, RecommendationResponse } from '~/types/api'

const route = useRoute()
const id = route.params.id as string
const api = useXcrsApi()
const { fill } = useBoard()

// --- Data: rendered on the server for shared links, then polled while explanations arrive (ADR-0018) ---
const { data, error } = await useAsyncData(`recommendation-${id}`, () => api.recommendation(id))
if (error.value && import.meta.server) {
  const event = useRequestEvent()
  if (event) setResponseStatus(event, 404)
}
const roles = computed(() => data.value?.roles ?? [])
const courses = computed(() => groupCourses(roles.value))
const pending = computed(() => roles.value.filter((r) => r.explanation_status === 'pending').length)
const topScore = computed(() => Math.max(...roles.value.map((r) => r.score), 1e-9))

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

useHead(() => ({ title: roles.value.length ? `${roles.value[0].role} and more · XCRS` : 'Your results · XCRS' }))

// --- Hover: a role lights up its courses and lines; a course lights up its roles ---
const hover = ref<{ kind: 'role' | 'course'; id: number } | null>(null)
const linked = (roleId: number, courseId: number) =>
  roles.value.some((r) => r.role_id === roleId && r.courses.some((c) => c.course_id === courseId))

function roleState(roleId: number) {
  if (!hover.value) return 'normal'
  if (hover.value.kind === 'role') return hover.value.id === roleId ? 'highlighted' : 'dimmed'
  return linked(roleId, hover.value.id) ? 'highlighted' : 'dimmed'
}
function courseState(courseId: number) {
  if (!hover.value) return 'normal'
  if (hover.value.kind === 'course') return hover.value.id === courseId ? 'highlighted' : 'dimmed'
  return linked(hover.value.id, courseId) ? 'highlighted' : 'dimmed'
}

// --- Connector lines between role cards and course cards (large screens) ---
const board = ref<HTMLElement | null>(null)
const roleEls = new Map<number, HTMLElement>()
const courseEls = new Map<number, HTMLElement>()
const paths = ref<{ d: string; color: string; roleId: number; courseId: number; start: number[]; end: number[] }[]>([])
const size = ref({ w: 0, h: 0 })

function setRef(map: Map<number, HTMLElement>, key: number) {
  return (el: unknown) => (el ? map.set(key, el as HTMLElement) : map.delete(key))
}

function layout() {
  const container = board.value
  if (!container || !window.matchMedia('(min-width: 1024px)').matches) {
    paths.value = []
    return
  }
  const box = container.getBoundingClientRect()
  size.value = { w: box.width, h: box.height }
  const next: typeof paths.value = []
  roles.value.forEach((role, index) => {
    const roleEl = roleEls.get(role.role_id)
    if (!roleEl) return
    const r = roleEl.getBoundingClientRect()
    const n = role.courses.length
    role.courses.forEach((c, j) => {
      const courseEl = courseEls.get(c.course_id)
      if (!courseEl) return
      const k = courses.value.find((g) => g.course_id === c.course_id)!
      const slot = k.links.findIndex((l) => l.roleId === role.role_id)
      const cr = courseEl.getBoundingClientRect()
      const x1 = r.right - box.left
      const y1 = r.top - box.top + Math.min(r.height / 2, 56) + (j - (n - 1) / 2) * 12
      const x2 = cr.left - box.left
      const y2 = cr.top - box.top + Math.min(cr.height / 2, 44) + (slot - (k.links.length - 1) / 2) * 12
      const mid = (x1 + x2) / 2
      next.push({
        d: `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`,
        color: roleColor(index).hex,
        roleId: role.role_id,
        courseId: c.course_id,
        start: [x1, y1],
        end: [x2, y2],
      })
    })
  })
  paths.value = next
}

let frame = 0
const schedule = () => {
  cancelAnimationFrame(frame)
  frame = requestAnimationFrame(layout)
}
// Card heights change as explanations arrive; observing the cards keeps the lines attached.
let observer: ResizeObserver | undefined
function observeCards() {
  if (!observer) return
  observer.disconnect()
  if (board.value) observer.observe(board.value)
  for (const el of [...roleEls.values(), ...courseEls.values()]) observer.observe(el)
  schedule()
}
watch(() => [board.value, roles.value.length, courses.value.length], observeCards, { flush: 'post' })
onMounted(() => {
  observer = new ResizeObserver(schedule)
  window.addEventListener('resize', schedule)
  observeCards()
})
onBeforeUnmount(() => {
  observer?.disconnect()
  window.removeEventListener('resize', schedule)
})

function lineState(p: { roleId: number; courseId: number }) {
  if (!hover.value) return 'normal'
  const on = hover.value.kind === 'role' ? hover.value.id === p.roleId : hover.value.id === p.courseId
  return on ? 'highlighted' : 'dimmed'
}

// --- Actions ---
const toast = useToast()
function editAnswers() {
  if (data.value?.input) fill(data.value.input)
  navigateTo('/')
}
async function copyLink() {
  try {
    await navigator.clipboard.writeText(window.location.href)
    toast.add({ title: 'Link copied', icon: 'i-heroicons-link', timeout: 2000 })
  } catch {
    toast.add({ title: 'Could not copy the link', color: 'amber' })
  }
}

const inputSummary = computed(() =>
  (Object.entries(data.value?.input ?? {}) as [Category, string[]][]).filter(([, items]) => items.length),
)
</script>

<template>
  <div class="mx-auto max-w-7xl px-4 pb-16 pt-8">
    <!-- Not found / error -->
    <div v-if="error || !data" class="mx-auto max-w-md py-20 text-center">
      <UIcon name="i-heroicons-magnifying-glass" class="mx-auto h-10 w-10 text-slate-300" />
      <h1 class="mt-3 text-xl font-semibold">We couldn't find these results</h1>
      <p class="mt-1 text-slate-500">The link may be wrong, or the server may be unavailable.</p>
      <UButton class="mt-6" to="/" label="Start a new search" icon="i-heroicons-arrow-left-20-solid" />
    </div>

    <template v-else>
      <div class="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p class="text-sm font-medium text-indigo-600">Your results</p>
          <h1 class="mt-1 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
            <template v-if="roles.length">Your top {{ roles.length === 1 ? 'role' : `${roles.length} roles` }} and the courses for each</template>
            <template v-else>No role matched yet</template>
          </h1>
        </div>
        <div class="flex gap-2">
          <UButton color="gray" variant="soft" icon="i-heroicons-pencil-square" label="Edit my answers" @click="editAnswers" />
          <UButton color="gray" variant="ghost" icon="i-heroicons-link" label="Copy link" @click="copyLink" />
        </div>
      </div>

      <div v-if="inputSummary.length" class="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
        <span class="text-slate-400">Based on</span>
        <span v-for="[category, items] in inputSummary" :key="category" class="flex flex-wrap items-center gap-1">
          <UIcon :name="CATEGORY_META[category].icon" class="h-3.5 w-3.5 text-slate-400" :aria-label="CATEGORY_META[category].title" />
          <span v-for="item in items" :key="item" class="rounded-full px-2 py-0.5 ring-1 ring-inset" :class="CATEGORY_META[category].chip">{{ item }}</span>
        </span>
      </div>

      <div v-if="pending" class="mt-4 inline-flex items-center gap-2 rounded-full bg-indigo-50 px-3 py-1 text-xs text-indigo-700" aria-live="polite">
        <UIcon name="i-heroicons-arrow-path" class="h-3.5 w-3.5 animate-spin" />
        Writing explanations: {{ roles.length - pending }} of {{ roles.length }} ready. The recommendations below are final.
      </div>

      <!-- Nothing matched -->
      <div v-if="!roles.length" class="mt-10 max-w-xl rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
        <h2 class="font-semibold">We couldn't connect your input to the career roadmaps</h2>
        <p class="mt-2 text-sm text-slate-600">
          Try naming specific technologies or subjects you've studied, like “Python”, “SQL”, “React” or “Linux”,
          and add a few you're curious about. The suggestions on the first page are a good place to start.
        </p>
        <UButton class="mt-4" label="Edit my answers" icon="i-heroicons-pencil-square" @click="editAnswers" />
      </div>

      <!-- Roles ↔ courses -->
      <div
        v-else
        ref="board"
        class="relative mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_5rem_minmax(0,1.15fr)] lg:gap-0"
      >
        <svg
          v-if="paths.length"
          class="pointer-events-none absolute inset-0 hidden lg:block"
          :width="size.w"
          :height="size.h"
          aria-hidden="true"
        >
          <g v-for="p in paths" :key="`${p.roleId}-${p.courseId}`" class="transition-opacity duration-200" :opacity="lineState(p) === 'dimmed' ? 0.1 : lineState(p) === 'highlighted' ? 1 : 0.55">
            <path :d="p.d" fill="none" :stroke="p.color" :stroke-width="lineState(p) === 'highlighted' ? 3 : 2" stroke-linecap="round" />
            <circle :cx="p.start[0]" :cy="p.start[1]" r="3.5" :fill="p.color" />
            <circle :cx="p.end[0]" :cy="p.end[1]" r="3.5" :fill="p.color" />
          </g>
        </svg>

        <section aria-label="Recommended roles" class="relative space-y-5 lg:pr-2">
          <h2 class="hidden text-xs font-semibold uppercase tracking-wide text-slate-400 lg:block">Career roles</h2>
          <div
            v-for="(role, index) in roles"
            :key="role.role_id"
            :ref="setRef(roleEls, role.role_id)"
            @mouseenter="hover = { kind: 'role', id: role.role_id }"
            @mouseleave="hover = null"
            @focusin="hover = { kind: 'role', id: role.role_id }"
            @focusout="hover = null"
          >
            <ResultsRoleCard
              :role="role"
              :index="index"
              :relative-fit="role.score / topScore"
              :request-id="data.request_id"
              :state="roleState(role.role_id)"
            />
          </div>
        </section>

        <div class="hidden lg:block" aria-hidden="true" />

        <section aria-label="Recommended courses" class="relative hidden space-y-3 lg:block lg:pl-2">
          <h2 class="text-xs font-semibold uppercase tracking-wide text-slate-400">Courses</h2>
          <div
            v-for="course in courses"
            :key="course.course_id"
            :ref="setRef(courseEls, course.course_id)"
            @mouseenter="hover = { kind: 'course', id: course.course_id }"
            @mouseleave="hover = null"
            @focusin="hover = { kind: 'course', id: course.course_id }"
            @focusout="hover = null"
          >
            <ResultsCourseCard
              :course="course"
              :request-id="data.request_id"
              :state="courseState(course.course_id)"
              :highlighted-role="hover?.kind === 'role' ? hover.id : null"
            />
          </div>
        </section>
      </div>
    </template>
  </div>
</template>
