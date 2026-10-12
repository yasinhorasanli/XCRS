<script setup lang="ts">
import { CATEGORIES, CATEGORY_META } from '~/composables/categories'
import { LEVEL_NAMES, PROFICIENCY_NAMES } from '~/composables/useXcrsApiV2'
import type { Category, Gap, ResourceSection, RoleResult, SkillRef, SkillToAdd } from '~/types/apiV2'

const props = defineProps<{ role: RoleResult; rank: number; recommendationId: string }>()
const emit = defineEmits<{ add: [items: SkillToAdd[]] }>()
const api = useXcrsApiV2()
const { onBoard } = useBoardV2()

/** Job titles to search for (ADR-0044): the learner's own first; each opens a LinkedIn Jobs search. */
const titles = computed(() => {
  const own = props.role.title_for_you
  return [...(own ? [own] : []), ...(props.role.job_titles ?? []).filter((t) => t !== own)]
})
const jobSearch = (title: string) => `https://www.linkedin.com/jobs/search/?keywords=${encodeURIComponent(title)}`

/** "45 min", "4 h 30 min", "about 12 h" (long playlists: the minutes don't matter). */
function duration(minutes: number): string {
  if (minutes < 60) return `${minutes} min`
  const h = Math.floor(minutes / 60)
  if (h >= 10) return `about ${Math.round(minutes / 60)} h`
  return minutes % 60 ? `${h} h ${minutes % 60} min` : `${h} h`
}

function clock(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = String(seconds % 60).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`
}

/** Where to start (ADR-0046): a chapter at its time, a playlist's video, or episode 1 of a playlist listed newest first. */
function sectionLabel(section: ResourceSection): string {
  if (section.kind === 'start') return `Start with: ${section.title}`
  if (section.kind === 'chapter' && section.start_seconds != null) return `Jump to ${clock(section.start_seconds)}: ${section.title}`
  return `Watch: ${section.title}`
}

const gapOnBoard = (g: Gap) => g.skills.some((s) => onBoard(s))
// "Add to Curious about": the Curious box's light bulb, in its colour.
const CURIOUS_BUTTON =
  'ml-1 grid h-6 w-6 place-items-center rounded-full bg-violet-50 dark:bg-violet-950/40 text-violet-700 dark:text-violet-200 ring-1 ring-inset ring-violet-200 dark:ring-violet-800 transition hover:bg-violet-100 dark:hover:bg-violet-900/50'
const CURIOUS_ICON = CATEGORY_META.curious.icon

/** An icon per resource type; YouTube videos and playlists get YouTube's own mark (in its red). */
const TYPE_ICONS: Record<string, string> = {
  course: 'i-heroicons-academic-cap',
  tutorial: 'i-heroicons-command-line',
  docs: 'i-heroicons-document-text',
  book: 'i-heroicons-book-open',
  video: 'i-heroicons-play-circle',
  playlist: 'i-heroicons-queue-list',
}
const isYouTube = (url: string) => /^https:\/\/(www\.|m\.)?(youtube\.com|youtu\.be)\//.test(url)
function typeIcon(type: string, url: string): { name: string; class: string } {
  if ((type === 'video' || type === 'playlist') && isYouTube(url)) return { name: 'i-simple-icons-youtube', class: 'text-[#FF0000]' }
  return { name: TYPE_ICONS[type] ?? 'i-heroicons-link', class: '' }
}


/** The answers to "Do you know it?", in the learner's words, mapped to the board's boxes. */
const KNOW_ANSWERS: Record<Category, string> = {
  liked: 'Yes, and I enjoyed it',
  neutral: 'Yes, no strong feelings',
  disliked: "Yes, but I didn't enjoy it",
  curious: 'No, but I want to learn it',
}
/** Level colours (2026-10-12): emerald getting darker as the level rises. */
const LEVEL_COLOURS = ['#6ee7b7', '#34d399', '#10b981', '#059669']
const levelColour = (i: number) => LEVEL_COLOURS[Math.min(i + (4 - props.role.levels.length), 3)]!

/** A basic the learner can confirm: into any box, at the level the role assumes (curious has no level). */
function basicMenu(g: Gap) {
  const into = (skill: SkillRef) =>
    CATEGORIES.map((category) => ({
      label: KNOW_ANSWERS[category],
      icon: CATEGORY_META[category].icon,
      onSelect: () => emit('add', [{ skill, category, proficiency: category === 'curious' ? undefined : g.need }]),
    }))
  const heading = (name: string) => ({ type: 'label' as const, label: `Do you know ${name}?` })
  return g.skills.map((s) => [heading(s.name), ...into(s)])
}

/** A skill to learn goes to "Curious"; with a choice ("Kotlin or Java"), the learner picks one. */
const gapMenu = (g: Gap) => [
  [
    { type: 'label' as const, label: 'Add to “Curious about”' },
    ...g.skills.map((skill) => ({ label: skill.name, onSelect: () => emit('add', [{ skill, category: 'curious' as Category }]) })),
  ],
]

/** The level bar (2026-10-12): reached levels full, the next one partly (how much of what it adds is on the board),
 * later ones empty. Results before engine v3.7 have no progress: reached levels full, the rest empty. */
const reachedIndex = computed(() => (props.role.level ? props.role.levels.findIndex((l) => l.id === props.role.level!.id) : -1))
const targetIndex = computed(() => props.role.levels.findIndex((l) => l.id === props.role.target_level.id))
function progress(i: number): number {
  const lv = props.role.levels[i]!
  if (lv.progress != null) return lv.progress
  return i <= reachedIndex.value ? 1 : 0
}
/** Skills of a reached level the learner didn't list, for the review under that level ("basics"). */
const reviewOf = (levelId: string) =>
  (props.role.basics ?? []).filter((g) => (g.level ?? props.role.level?.id) === levelId)
const openReview = ref<string | null>(null)

/** The next level is where the recommendations start: a tap on it scrolls to "Start learning" (or the skills to
 * learn when there are no resources) and frames it for a moment. */
const isNext = (i: number) => i === targetIndex.value && i !== reachedIndex.value
const learnRef = ref<HTMLElement>()
const gapsRef = ref<HTMLElement>()
function goToLearning() {
  const el = learnRef.value ?? gapsRef.value
  if (!el) return
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  el.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'center' })
  el.classList.remove('flash-frame')
  void el.offsetWidth // restart the animation on a second tap
  el.classList.add('flash-frame')
}
const toggleReview = (id: string) => (openReview.value = openReview.value === id ? null : id)

function addAllBasics(levelId: string) {
  const items = reviewOf(levelId)
    .filter((g) => !gapOnBoard(g))
    .map((g) => ({ skill: g.skills[0]!, category: 'neutral' as Category, proficiency: g.need }))
  if (items.length) emit('add', items)
}

const levelName = (id: string, title?: string | null) => title ?? LEVEL_NAMES[id] ?? id
// What the learner sees (ADR-0052): the share of the role's skills they enjoy or are curious about; results made
// before engine v3.3 only have the ranking interest (a share of the learner's input).
const interestShown = computed(() => props.role.interest_in_role ?? props.role.interest)
const pct = (x: number) => `${Math.round(Math.max(0, Math.min(1, x)) * 100)}%`
const stages = computed(() => {
  const out: { stage: string; gaps: RoleResult['gaps'] }[] = []
  for (const g of props.role.gaps) {
    const last = out.at(-1)
    if (last && last.stage === g.stage) last.gaps.push(g)
    else out.push({ stage: g.stage, gaps: [g] })
  }
  return out
})

const vote = ref<1 | -1 | null>(null)
async function send(rating: 1 | -1) {
  if (vote.value) return
  try {
    await api.feedback(props.recommendationId, { role: props.role.id, rating })
    vote.value = rating
  } catch {
    // feedback is optional
  }
}
</script>

<template>
  <article class="rounded-2xl bg-default p-5 shadow-xs ring-1 ring-default">
    <header class="flex flex-wrap items-start gap-3">
      <span class="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-indigo-600 dark:bg-indigo-700 text-sm font-bold text-white">{{ rank }}</span>
      <div class="min-w-0 flex-1">
        <h2 class="text-lg font-semibold">{{ role.name }}</h2>
        <p class="text-sm text-muted">
          <template v-if="role.level">Estimated level: <span class="font-medium text-default">{{ levelName(role.level.id, role.level.title) }}</span></template>
          <template v-else>Start at <span class="font-medium text-default">{{ levelName(role.target_level.id, role.target_level.title) }}</span></template>
        </p>
      </div>
      <div class="flex items-center gap-1 print:hidden" aria-label="Was this role a good suggestion?">
        <UButton size="xs" color="neutral" :variant="vote === 1 ? 'solid' : 'ghost'" icon="i-heroicons-hand-thumb-up" aria-label="Good suggestion" :disabled="!!vote" @click="send(1)" />
        <UButton size="xs" color="neutral" :variant="vote === -1 ? 'solid' : 'ghost'" icon="i-heroicons-hand-thumb-down" aria-label="Not for me" :disabled="!!vote" @click="send(-1)" />
      </div>
    </header>

    <div v-if="role.explanation_status === 'pending'" class="mt-3 flex items-center gap-2 text-sm text-muted" aria-live="polite">
      <UIcon name="i-heroicons-arrow-path" class="h-4 w-4 animate-spin" />
      Writing an explanation for you…
    </div>
    <!-- A readable line length, now that the cards are as wide as the page. -->
    <div v-else-if="role.explanation" class="mt-3 max-w-4xl space-y-2 text-[15px] leading-relaxed text-default" aria-live="polite">
      <p>{{ role.explanation }}</p>
    </div>

    <div class="mt-4 grid gap-x-6 gap-y-4 sm:grid-cols-2">
      <!-- The level bar: one segment per level of the role. Same three rows as Interest, so the bars line up. -->
      <div>
        <p class="text-xs text-muted">Your level in this role</p>
        <ol class="mt-1 grid gap-1" :style="{ gridTemplateColumns: `repeat(${role.levels.length}, minmax(0, 1fr))` }" aria-label="Levels">
          <li v-for="(lv, i) in role.levels" :key="lv.id" class="min-w-0">
            <component
              :is="isNext(i) ? 'button' : 'div'"
              :type="isNext(i) ? 'button' : undefined"
              class="block w-full rounded-md text-left"
              :class="isNext(i) ? 'group cursor-pointer' : ''"
              :title="isNext(i) ? `What to learn for ${levelName(lv.id, lv.title)}` : undefined"
              @click="isNext(i) && goToLearning()"
            >
            <span
              class="flex items-center gap-0.5 truncate text-[11px]"
              :class="i === reachedIndex ? 'font-semibold text-default' : isNext(i) ? 'font-medium text-emerald-700 group-hover:underline dark:text-emerald-300' : 'text-dimmed'"
            >{{ levelName(lv.id, lv.title) }}<UIcon v-if="isNext(i)" name="i-heroicons-arrow-down-20-solid" class="h-3 w-3 shrink-0 print:hidden" /></span>
            <span
              class="liquid-track mt-1"
              role="img"
              :aria-label="i <= reachedIndex ? 'reached' : i === targetIndex ? 'next level, partly shown' : 'later'"
            >
              <span
                v-if="progress(i) > 0"
                class="liquid-fill"
                :class="i > reachedIndex ? 'flowing opacity-80' : ''"
                :style="{ width: `${Math.max(8, Math.round(progress(i) * 100))}%`, '--c': levelColour(i) }"
              />
            </span>
            </component>
            <!-- Easy to spot: the skills this level assumes that the board doesn't show. -->
            <button
              v-if="i <= reachedIndex && reviewOf(lv.id).length"
              type="button"
              class="mt-1.5 inline-flex max-w-full items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-800 ring-1 ring-inset ring-amber-200 transition hover:bg-amber-100 dark:bg-amber-950/40 dark:text-amber-200 dark:ring-amber-800 dark:hover:bg-amber-900/50 print:hidden"
              :class="openReview === lv.id ? 'ring-2 ring-amber-400 dark:ring-amber-500' : ''"
              :aria-expanded="openReview === lv.id"
              :title="`Skills ${levelName(lv.id, lv.title)} assumes that aren't on your board`"
              @click="toggleReview(lv.id)"
            >
              <UIcon name="i-heroicons-clipboard-document-check" class="h-3.5 w-3.5 shrink-0" />
              <span class="truncate">{{ reviewOf(lv.id).length }} to check</span>
              <UIcon :name="openReview === lv.id ? 'i-heroicons-chevron-up-20-solid' : 'i-heroicons-chevron-down-20-solid'" class="h-3.5 w-3.5 shrink-0" />
            </button>
          </li>
        </ol>
      </div>
      <!-- On phones Interest comes first, so the level bar sits right above its review. -->
      <div class="order-first sm:order-none">
        <p class="text-xs text-muted">Interest</p>
        <p class="mt-1 truncate text-[11px] text-dimmed">
          <span class="font-semibold tabular-nums text-violet-700 dark:text-violet-300">{{ pct(interestShown) }}</span>
          of the role's skills you enjoy or are curious about
        </p>
        <span class="liquid-track mt-1" role="img" :aria-label="`Interest ${pct(interestShown)}`">
          <span v-if="interestShown > 0" class="liquid-fill" :style="{ width: `max(0.75rem, ${pct(interestShown)})`, '--c': '#8b5cf6' }" />
        </span>
      </div>
    </div>

    <!-- Review of a reached level: its skills you didn't list. Open by "Review" (on paper, all of them). -->
    <template v-for="(lv, i) in role.levels" :key="`review-${lv.id}`">
      <div
        v-if="i <= reachedIndex && reviewOf(lv.id).length"
        class="mt-3 rounded-xl bg-muted/60 px-3 py-2.5 text-sm ring-1 ring-inset ring-default"
        :class="openReview === lv.id ? '' : 'hidden print:block'"
      >
        <p class="text-xs text-muted">
          <span class="font-medium text-default">{{ levelName(lv.id, lv.title) }}</span> assumes these skills, but they
          aren't on your board. Do you know them? <span class="print:hidden">Tap one to answer, or
          <button type="button" class="font-medium text-indigo-600 dark:text-indigo-300 hover:underline" @click="addAllBasics(lv.id)">I know all of these</button>.</span>
        </p>
        <ul class="mt-2 flex flex-wrap gap-1.5">
          <li v-for="g in reviewOf(lv.id)" :key="g.skills.map((x) => x.id).join('|')">
            <span v-if="gapOnBoard(g)" class="inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs text-dimmed ring-1 ring-inset ring-default">
              <UIcon name="i-heroicons-check-20-solid" class="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-300" />
              {{ g.skills.map((x) => x.name).join(' or ') }}
            </span>
            <UDropdownMenu v-else :items="basicMenu(g)">
              <button type="button" class="inline-flex items-center gap-1 rounded-lg bg-default px-2.5 py-1 text-xs text-toned ring-1 ring-inset ring-default transition hover:bg-muted hover:ring-indigo-300 dark:hover:ring-indigo-800">
                {{ g.skills.map((x) => x.name).join(' or ') }}
                <span class="text-dimmed">· {{ PROFICIENCY_NAMES[g.need]?.toLowerCase() }}</span>
                <UIcon name="i-heroicons-plus-20-solid" class="h-3.5 w-3.5 text-dimmed print:hidden" />
              </button>
            </UDropdownMenu>
          </li>
        </ul>
      </div>
    </template>

    <div v-if="role.because.length" class="mt-4">
      <h3 class="text-xs font-medium uppercase tracking-wide text-muted">Because of</h3>
      <div class="mt-1.5 flex flex-wrap gap-1.5">
        <span v-for="b in role.because" :key="b.id" class="rounded-full px-2.5 py-0.5 text-xs ring-1 ring-inset" :class="CATEGORY_META[b.category].chip">
          {{ b.name }} <span class="opacity-90">· {{ CATEGORY_META[b.category].short.toLowerCase() }}</span>
        </span>
      </div>
    </div>

    <div v-if="titles.length" class="mt-4">
      <h3 class="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted">
        Job titles to search on
        <span class="inline-flex items-center gap-1 normal-case tracking-normal text-[#0A66C2] dark:text-sky-300"><UIcon name="i-simple-icons-linkedin" class="h-3.5 w-3.5" />LinkedIn</span>
      </h3>
      <ul class="mt-1.5 flex flex-wrap gap-1.5">
        <li v-for="(t, i) in titles" :key="t">
          <a
            :href="jobSearch(t)"
            target="_blank"
            rel="noopener"
            class="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs ring-1 ring-inset transition hover:ring-indigo-300 dark:hover:ring-indigo-800"
            :class="i === 0 && role.title_for_you ? 'bg-indigo-50 dark:bg-indigo-950/40 font-medium text-indigo-800 dark:text-indigo-200 ring-indigo-200 dark:ring-indigo-800' : 'text-default ring-default'"
            :title="`Search LinkedIn Jobs for “${t}”`"
          >
            <UIcon v-if="i === 0 && role.title_for_you" name="i-heroicons-sparkles" class="h-3.5 w-3.5" />
            {{ t }}
            <UIcon name="i-heroicons-arrow-top-right-on-square" class="h-3 w-3 opacity-50" />
          </a>
        </li>
      </ul>
    </div>

    <div v-if="role.resources?.length" ref="learnRef" class="mt-4">
      <h3 class="text-xs font-medium uppercase tracking-wide text-muted">Start learning</h3>
      <ul class="mt-1.5 grid gap-2 sm:grid-cols-3">
        <li v-for="res in role.resources" :key="res.id">
          <!-- The whole card opens the resource (a stretched link); a section link sits above it. -->
          <div class="relative flex h-full flex-col rounded-xl p-3 ring-1 ring-inset ring-default transition hover:-translate-y-px hover:ring-indigo-300 dark:hover:ring-indigo-800">
            <span class="flex items-center gap-1.5 text-[11px] text-muted">
              <span class="inline-flex items-center gap-1 rounded bg-elevated px-1.5 py-0.5 capitalize">
                <UIcon :name="typeIcon(res.type, res.url).name" class="h-3.5 w-3.5" :class="typeIcon(res.type, res.url).class" />{{ res.type }}
              </span>
              <span v-if="res.free" class="rounded bg-emerald-50 dark:bg-emerald-950/40 px-1.5 py-0.5 text-emerald-700 dark:text-emerald-200">Free</span>
              <span v-if="res.duration_minutes" class="shrink-0 tabular-nums">{{ duration(res.duration_minutes) }}</span>
              <span class="truncate">{{ res.provider }}</span>
            </span>
            <a :href="res.url" target="_blank" rel="noopener" class="mt-1 text-sm font-medium leading-snug after:absolute after:inset-0 after:rounded-xl">{{ res.title }}</a>
            <a
              v-if="res.section"
              :href="res.section.url"
              target="_blank"
              rel="noopener"
              class="relative z-10 mt-1 flex items-start gap-1 text-[11px] text-indigo-700 hover:underline dark:text-indigo-300"
            >
              <UIcon name="i-heroicons-play-circle" class="mt-px h-3.5 w-3.5 shrink-0" />
              <span>{{ sectionLabel(res.section) }}</span>
            </a>
            <span class="mt-auto pt-1 text-[11px] text-dimmed">For {{ res.skills.map((s) => s.name).join(', ') }}</span>
            <span class="hidden break-all pt-0.5 text-[10px] text-dimmed print:block">{{ res.url }}</span>
          </div>
        </li>
      </ul>
    </div>

    <div v-if="role.gaps.length" ref="gapsRef" class="mt-4">
      <h3 class="text-xs font-medium uppercase tracking-wide text-muted">
        <template v-if="role.level?.id === role.target_level.id">Still to cover at {{ levelName(role.target_level.id, role.target_level.title) }}</template>
        <template v-else>To reach {{ levelName(role.target_level.id, role.target_level.title) }}</template>: {{ role.gaps_total }} {{ role.gaps_total === 1 ? 'skill' : 'skills' }} to learn<span v-if="role.gaps_total > role.gaps.length">, first {{ role.gaps.length }}</span>
      </h3>
      <p v-if="rank === 1" class="mt-1 flex flex-wrap items-center gap-x-1 text-xs text-muted print:hidden">
        Tap <UIcon :name="CURIOUS_ICON" class="h-4 w-4 text-violet-600 dark:text-violet-300" aria-label="the light bulb" /> to add a skill to
        “Curious about” on your board, then update your results.
      </p>
      <div v-for="s in stages" :key="s.stage" class="mt-2">
        <p class="text-xs text-dimmed">{{ s.stage }}</p>
        <ul class="mt-1 flex flex-wrap gap-1.5">
          <li v-for="g in s.gaps" :key="g.skills.map((x) => x.id).join('|')" class="flex items-center gap-1 rounded-lg bg-muted py-1 pl-2.5 pr-1 text-sm ring-1 ring-inset ring-default">
            <span>
              {{ g.skills.map((x) => x.name).join(' or ') }}
              <span class="text-[11px] text-muted">· {{ PROFICIENCY_NAMES[g.need]?.toLowerCase() }}<template v-if="g.have"> (you: {{ PROFICIENCY_NAMES[g.have]?.toLowerCase() }})</template></span>
            </span>
            <UIcon v-if="gapOnBoard(g)" name="i-heroicons-check-20-solid" class="mx-1 h-4 w-4 print:hidden text-emerald-600 dark:text-emerald-300" aria-label="On your board" />
            <button
              v-else-if="g.skills.length === 1"
              type="button"
              :class="[CURIOUS_BUTTON, 'print:hidden']"
              :aria-label="`Add ${g.skills[0]!.name} to Curious about`"
              :title="`Add ${g.skills[0]!.name} to “Curious about”`"
              @click="emit('add', [{ skill: g.skills[0]!, category: 'curious' }])"
            >
              <UIcon :name="CURIOUS_ICON" class="h-3.5 w-3.5" />
            </button>
            <UDropdownMenu v-else :items="gapMenu(g)">
              <button
                type="button"
                :class="[CURIOUS_BUTTON, 'print:hidden']"
                :aria-label="`Add one of ${g.skills.map((x) => x.name).join(', ')} to Curious about`"
                title="Add one to “Curious about”"
              >
                <UIcon :name="CURIOUS_ICON" class="h-3.5 w-3.5" />
              </button>
            </UDropdownMenu>
          </li>
        </ul>
      </div>
    </div>

  </article>
</template>
