<script setup lang="ts">
// "Explainable [Career / Course / Concept] Recommendation System". The middle word sits on the faces of an
// invisible prism (CSS 3D) that turns one face at a time with an impact-and-recoil motion (Web Animations API).
// Only the text is drawn: no faces, edges or backgrounds. The slot takes the width of the word showing, and
// eases to the next word's width while the prism turns.
export interface HeadlineWord {
  text: string
  /** Literal Tailwind classes (Tailwind only keeps classes it can see in the source). */
  class: string
}

const props = withDefaults(defineProps<{
  before?: string
  after?: string
  words?: HeadlineWord[]
  /** Pause between turns, ms. */
  interval?: number
  /** Length of one turn, ms. */
  duration?: number
}>(), {
  before: 'Explainable',
  after: 'Recommendation System',
  // Board category colours: enjoyed, didn't enjoy, curious (composables/categories.ts), as in the logo.
  words: () => [
    { text: 'Career', class: 'text-emerald-600 dark:text-emerald-400' },
    { text: 'Course', class: 'text-rose-600 dark:text-rose-400' },
    { text: 'Concept', class: 'text-violet-600 dark:text-violet-400' },
  ],
  interval: 1500,
  duration: 700,
})

const step = computed(() => 360 / props.words.length)
const index = ref(0)
const prism = ref<HTMLElement>()
const slot = ref<HTMLElement>()
const label = computed(() => {
  const w = props.words.map(x => x.text)
  const list = w.length > 1 ? `${w.slice(0, -1).join(', ')} and ${w.at(-1)}` : w.join('')
  return `${props.before} ${list} ${props.after}`
})

// One turn as progress over time (1 = the next face): an instant launch at full speed, a hard brake that
// overshoots by 8°, a recoil to 4° short, a 1° correction, then rest. Sampled as a CSS linear() curve, which can
// leave 0..1; browsers without linear() get one overshooting cubic-bezier.
function turnEasing(s: number): string {
  const curve: [number, number][] = [
    [0, 0], [0.03, 0.07], [0.06, 0.17], [0.1, 0.31], [0.14, 0.45], [0.18, 0.59], [0.22, 0.73], [0.26, 0.86],
    [0.29, 0.95], [0.31, 1 + 2 / s], [0.335, 1 + 6.5 / s], [0.36, 1 + 8 / s], // hard brake, 8° past
    [0.4, 1 + 7 / s], [0.45, 1 + 2.5 / s], [0.5, 1 - 2.5 / s], [0.55, 1 - 4 / s], // recoil to 4° short
    [0.6, 1 - 3 / s], [0.66, 1 - 0.5 / s], [0.72, 1 + 1 / s], [0.79, 1 + 0.7 / s], [0.87, 1 + 0.1 / s], [1, 1],
  ]
  return `linear(${curve.map(([t, p]) => `${+p.toFixed(4)} ${+(t * 100).toFixed(1)}%`).join(', ')})`
}
const FALLBACK_EASING = 'cubic-bezier(0.2, 0.9, 0.3, 1.25)'

// The slot's width follows the prism: most of the change during the fast part, settled by the brake.
const WIDTH_EASING = 'cubic-bezier(0.3, 0.6, 0.35, 1)'
const WIDTH_SHARE = 0.34

// Motion blur while the prism is fastest: a soft blur plus ghost copies smeared along the direction of travel.
const SMEAR = '0 -0.07em 0 color-mix(in srgb, currentColor 30%, transparent), 0 0.07em 0 color-mix(in srgb, currentColor 30%, transparent)'
const BLUR_FRAMES: Keyframe[] = [
  { offset: 0, filter: 'blur(0)', textShadow: 'none' },
  { offset: 0.04, filter: 'blur(0.035em)', textShadow: SMEAR },
  { offset: 0.24, filter: 'blur(0.045em)', textShadow: SMEAR },
  { offset: 0.3, filter: 'blur(0.012em)', textShadow: 'none' },
  { offset: 0.34, filter: 'blur(0)', textShadow: 'none' },
]

let timer: ReturnType<typeof setTimeout> | undefined
let running: Animation[] = []
let stopped = false

function schedule() {
  clearTimeout(timer)
  timer = setTimeout(turn, props.interval)
}

async function turn() {
  timer = undefined
  const el = prism.value
  if (stopped || !el || document.hidden) return
  const next = (index.value + 1) % props.words.length
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  try {
    if (reduce) {
      // No spinning: a quick fade out and in.
      const fade = { duration: 180, easing: 'ease-in-out' }
      running = [slot.value!.animate([{ opacity: 1 }, { opacity: 0 }], { ...fade, fill: 'forwards' })]
      await running[0]!.finished
      index.value = next
      await nextTick()
      running[0]!.cancel()
      running = [slot.value!.animate([{ opacity: 0 }, { opacity: 1 }], fade)]
      await running[0]!.finished
    }
    else {
      const from = -index.value * step.value
      const easing = CSS.supports('animation-timing-function', 'linear(0, 1)') ? turnEasing(step.value) : FALLBACK_EASING
      const turnAnim = el.animate(
        [{ transform: `rotateX(${from}deg)` }, { transform: `rotateX(${from - step.value}deg)` }],
        { duration: props.duration, easing, fill: 'forwards' },
      )
      const faces = Array.from(el.children) as HTMLElement[]
      // offsetWidth is the layout width, unaffected by the faces' rotation.
      const widthAnim = slot.value!.animate(
        [
          { width: `${slot.value!.offsetWidth}px`, easing: WIDTH_EASING },
          { width: `${faces[next]!.offsetWidth}px`, offset: WIDTH_SHARE },
          { width: `${faces[next]!.offsetWidth}px` },
        ],
        { duration: props.duration, fill: 'forwards' },
      )
      running = [turnAnim, widthAnim, ...faces.map(f => f.animate(BLUR_FRAMES, { duration: props.duration }))]
      await turnAnim.finished
      // Hand the final angle back to the template, then drop the animation (no stacked fill-forwards).
      index.value = next
      await nextTick()
      turnAnim.cancel()
      widthAnim.cancel() // the sizer now holds the new word
    }
  }
  catch {
    return // cancelled while unmounting
  }
  running = []
  if (!stopped) schedule()
}

function onVisibility() {
  if (document.hidden) {
    clearTimeout(timer)
    timer = undefined
  }
  else if (!timer && !running.length) {
    schedule()
  }
}

onMounted(() => {
  document.addEventListener('visibilitychange', onVisibility)
  schedule()
})

onBeforeUnmount(() => {
  stopped = true
  clearTimeout(timer)
  document.removeEventListener('visibilitychange', onVisibility)
  running.forEach(a => a.cancel())
})
</script>

<template>
  <p class="text-balance text-slate-900 dark:text-slate-50">
    <span class="sr-only">{{ label }}</span>
    <span aria-hidden="true">{{ before }}</span>
    {{ ' ' }}
    <span ref="slot" class="rh-slot" aria-hidden="true">
      <!-- Invisible copy of the word showing: gives the slot its width and baseline. -->
      <span class="rh-sizer">{{ words[index]?.text }}</span>
      <span class="rh-stage">
        <span class="rh-depth">
          <span ref="prism" class="rh-prism" :style="{ transform: `rotateX(${-index * step}deg)` }">
            <span
              v-for="(w, i) in words"
              :key="w.text"
              class="rh-face"
              :class="w.class"
              :style="{ transform: `rotateX(${i * step}deg) translateZ(var(--rh-depth))` }"
            >{{ w.text }}</span>
          </span>
        </span>
      </span>
    </span>
    {{ ' ' }}
    <span aria-hidden="true">{{ after }}</span>
  </p>
</template>

<style scoped>
.rh-slot {
  --rh-depth: 0.42em; /* distance from the prism's axis to each face */
  position: relative;
  display: inline-block;
  perspective: 5em;
  transform-style: preserve-3d;
  white-space: nowrap;
}
.rh-sizer {
  visibility: hidden;
}
/* The prism floats over the sizer, centred on it; mid-turn, a wider word spills out evenly on both sides
   (grid centring would push the overflow to one side). Pulled back by its depth, so the front face sits exactly
   where flat text would (no scaling). */
.rh-stage {
  position: absolute;
  inset: 0;
  transform-style: preserve-3d;
}
.rh-depth {
  position: absolute;
  top: 0;
  left: 50%;
  display: grid;
  transform-style: preserve-3d;
  transform: translateX(-50%) translateZ(calc(-1 * var(--rh-depth)));
}
.rh-prism {
  display: grid;
  justify-items: center;
  transform-style: preserve-3d;
}
.rh-face {
  grid-area: 1 / 1;
  backface-visibility: hidden;
  -webkit-backface-visibility: hidden;
}
</style>
