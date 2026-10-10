<script setup lang="ts">
import { CATEGORY_META } from '~/composables/categories'
import { MAX_CHIPS } from '~/composables/useBoardV2'
import type { Category } from '~/types/apiV2'
import type { SkillSuggestion } from '~/types/apiV2'

const props = defineProps<{ category: Category }>()
const { active, byCategory, add, remove, rate, onBoard } = useBoardV2()
const api = useXcrsApiV2()
const toast = useToast()

const meta = computed(() => CATEGORY_META[props.category])
const chips = computed(() => byCategory(props.category))
const isActive = computed(() => active.value === props.category)

function place(label: string, skill?: SkillSuggestion) {
  if (add(label, props.category, skill) === 'full') {
    toast.add({ title: 'The board is full', description: `Up to ${MAX_CHIPS} skills in total.`, color: 'warning' })
  }
}

// Typing searches the catalog; picking an option adds that skill, Enter without one adds your own words.
const text = ref('')
const options = ref<SkillSuggestion[]>([])
const highlighted = ref(-1)
const open = ref(false)
let timer: ReturnType<typeof setTimeout> | undefined
let latest = 0

watch(text, (value) => {
  clearTimeout(timer)
  const q = value.split(',').at(-1)!.trim()
  if (q.length < 2) {
    options.value = []
    return
  }
  timer = setTimeout(async () => {
    const call = ++latest
    try {
      const { skills } = await api.searchSkills(q, 12)
      if (call === latest) {
        options.value = skills.filter((s) => !onBoard(s)).slice(0, 7) // skills already on the board aren't offered again
        highlighted.value = -1
        open.value = true
      }
    } catch {
      options.value = []
    }
  }, 160)
})

function commit(option?: SkillSuggestion) {
  if (option) place(option.name, option)
  else text.value.split(',').forEach((l) => place(l))
  text.value = ''
  options.value = []
  open.value = false
}

function onBlur() {
  setTimeout(() => (open.value = false), 150) // let a click on an option land first
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'ArrowDown' && options.value.length) {
    event.preventDefault()
    open.value = true
    highlighted.value = (highlighted.value + 1) % options.value.length
  } else if (event.key === 'ArrowUp' && options.value.length) {
    event.preventDefault()
    highlighted.value = (highlighted.value - 1 + options.value.length) % options.value.length
  } else if (event.key === 'Enter') {
    event.preventDefault()
    commit(open.value && highlighted.value >= 0 ? options.value[highlighted.value] : undefined)
  } else if (event.key === 'Escape') {
    open.value = false
  }
}
</script>

<template>
  <section
    class="relative flex min-h-[11rem] flex-col rounded-2xl bg-default p-3 shadow-xs ring-1 transition"
    :class="isActive ? `ring-2 ${meta.ring}` : 'ring-default'"
    :aria-label="meta.title"
  >
    <button type="button" class="flex items-start gap-2.5 rounded-lg p-1 text-left" :aria-pressed="isActive" @click="active = category">
      <span class="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg" :class="meta.chip">
        <UIcon :name="meta.icon" class="h-4 w-4" />
      </span>
      <span class="min-w-0 flex-1">
        <span class="flex items-center gap-2">
          <span class="font-semibold">{{ meta.title }}</span>
          <span v-if="chips.length" class="text-xs tabular-nums text-dimmed">{{ chips.length }}</span>
          <span v-if="isActive" class="ml-auto rounded-full bg-slate-900 dark:bg-slate-700 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-white">Selected</span>
        </span>
        <span class="block text-xs text-muted">
          {{ meta.hint }}<template v-if="category !== 'curious'">. Dots: how well you know it (optional)</template>
        </span>
      </span>
    </button>

    <div class="mt-2 flex flex-1 flex-wrap content-start gap-1.5 px-1">
      <V2ChipV2 v-for="c in chips" :key="c.key" :chip="c" @remove="remove(c.key)" @rate="(n) => rate(c.key, n)" />
      <p v-if="!chips.length" class="w-full rounded-xl border border-dashed border-default px-3 py-4 text-center text-xs text-dimmed">
        Search the catalog or type your own words
      </p>
    </div>

    <div class="relative mt-2">
      <input
        v-model="text"
        type="text"
        maxlength="300"
        class="w-full rounded-lg border-0 bg-muted px-3 py-2 text-sm ring-1 ring-inset ring-default placeholder:text-dimmed focus:bg-default focus:outline-hidden focus:ring-2 focus:ring-indigo-400 dark:focus:ring-indigo-400"
        placeholder="Search skills, or type and press Enter…"
        :aria-label="`Add to ${meta.title}`"
        role="combobox"
        :aria-expanded="open && options.length > 0"
        autocomplete="off"
        @focus="active = category"
        @keydown="onKeydown"
        @blur="onBlur"
      >
      <ul v-if="open && options.length" class="absolute inset-x-0 top-full z-20 mt-1 max-h-64 overflow-auto rounded-xl bg-default py-1 text-sm shadow-lg ring-1 ring-default" role="listbox">
        <li
          v-for="(o, i) in options"
          :key="o.id"
          role="option"
          :aria-selected="i === highlighted"
          class="flex cursor-pointer items-center justify-between gap-3 px-3 py-1.5"
          :class="i === highlighted ? 'bg-indigo-50 dark:bg-indigo-950/40' : 'hover:bg-muted'"
          @mousedown.prevent="commit(o)"
        >
          <span class="truncate">{{ o.name }}</span>
          <span class="shrink-0 text-[11px] text-dimmed">{{ o.kind }}</span>
        </li>
      </ul>
    </div>
  </section>
</template>
