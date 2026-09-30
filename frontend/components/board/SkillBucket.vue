<script setup lang="ts">
import type { Category, KnowledgeUnit } from '~/types/api'

const props = defineProps<{ category: Category }>()

const { board, active, add, remove } = useBoard()
const api = useXcrsApi()
const toast = useToast()

const meta = computed(() => CATEGORY_META[props.category])
const skills = computed(() => board.value[props.category])
const isActive = computed(() => active.value === props.category)

// --- Drag and drop ---
const dragDepth = ref(0) // dragenter/leave fire for children too; count them
const isOver = computed(() => dragDepth.value > 0)

function onDrop(event: DragEvent) {
  dragDepth.value = 0
  const raw = event.dataTransfer?.getData(DRAG_TYPE)
  const label = raw ? (JSON.parse(raw).label as string) : event.dataTransfer?.getData('text/plain')
  if (label) place(label)
}

function place(label: string) {
  const outcome = add(label, props.category)
  if (outcome === 'full') {
    toast.add({ title: `“${meta.value.title}” is full`, description: `Up to ${MAX_PER_CATEGORY} items per box.`, color: 'amber' })
  }
}

// --- Typing, with suggestions from the catalog ---
const text = ref('')
const options = ref<KnowledgeUnit[]>([])
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
      const { units } = await api.search(q, 7)
      if (call === latest) {
        options.value = units
        highlighted.value = -1
        open.value = true
      }
    } catch {
      options.value = []
    }
  }, 180)
})

function commit(label?: string) {
  // "Java, SQL, Docker" adds three skills
  const labels = label ? [label] : text.value.split(',')
  labels.forEach((l) => place(l))
  text.value = ''
  options.value = []
  open.value = false
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
    commit(open.value && highlighted.value >= 0 ? options.value[highlighted.value].label : undefined)
  } else if (event.key === 'Escape') {
    open.value = false
  } else if (event.key === 'Backspace' && !text.value && skills.value.length) {
    remove(skills.value.at(-1)!)
  }
}

function onBlur() {
  setTimeout(() => (open.value = false), 150) // let a click on an option land first
}
</script>

<template>
  <section
    class="relative flex min-h-[11rem] flex-col rounded-2xl bg-white p-3 shadow-sm ring-1 transition"
    :class="[
      isOver ? `ring-2 ${meta.ring} ${meta.soft} scale-[1.01]` : isActive ? `ring-2 ${meta.ring}` : 'ring-slate-200',
    ]"
    :aria-label="meta.title"
    @dragenter.prevent="dragDepth++"
    @dragleave="dragDepth = Math.max(0, dragDepth - 1)"
    @dragover.prevent
    @drop.prevent="onDrop"
  >
    <button
      type="button"
      class="flex items-start gap-2.5 rounded-lg p-1 text-left"
      :aria-pressed="isActive"
      :title="`Clicking a suggestion adds it here`"
      @click="active = category"
    >
      <span class="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg" :class="meta.chip">
        <UIcon :name="meta.icon" class="h-4 w-4" />
      </span>
      <span class="min-w-0 flex-1">
        <span class="flex items-center gap-2">
          <span class="font-semibold">{{ meta.title }}</span>
          <span v-if="skills.length" class="text-xs tabular-nums text-slate-400">{{ skills.length }}</span>
          <span v-if="isActive" class="ml-auto rounded-full bg-slate-900 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-white">
            Selected
          </span>
        </span>
        <span class="block text-xs text-slate-500">{{ meta.hint }}</span>
      </span>
    </button>

    <div class="mt-2 flex flex-1 flex-wrap content-start gap-1.5 px-1">
      <BoardSkillChip v-for="s in skills" :key="s" :label="s" :category="category" @remove="remove(s)" />
      <p v-if="!skills.length" class="w-full rounded-xl border border-dashed border-slate-200 px-3 py-4 text-center text-xs text-slate-400">
        Drop skills here
      </p>
    </div>

    <div class="relative mt-2">
      <input
        v-model="text"
        type="text"
        :maxlength="MAX_LENGTH * 3"
        class="w-full rounded-lg border-0 bg-slate-50 px-3 py-2 text-sm ring-1 ring-inset ring-slate-200 placeholder:text-slate-400 focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-400"
        :placeholder="`Type and press Enter…`"
        :aria-label="`Add to ${meta.title}`"
        role="combobox"
        :aria-expanded="open && options.length > 0"
        autocomplete="off"
        @focus="active = category"
        @keydown="onKeydown"
        @blur="onBlur"
      >
      <ul
        v-if="open && options.length"
        class="absolute inset-x-0 top-full z-20 mt-1 max-h-64 overflow-auto rounded-xl bg-white py-1 text-sm shadow-lg ring-1 ring-slate-200"
        role="listbox"
      >
        <li
          v-for="(o, i) in options"
          :key="o.label"
          role="option"
          :aria-selected="i === highlighted"
          class="flex cursor-pointer items-center justify-between gap-3 px-3 py-1.5"
          :class="i === highlighted ? 'bg-indigo-50' : 'hover:bg-slate-50'"
          @mousedown.prevent="commit(o.label)"
        >
          <span class="truncate">{{ o.label }}</span>
          <span v-if="o.roles.length" class="shrink-0 text-[11px] text-slate-400">
            {{ o.roles.length === 1 ? o.roles[0] : `${o.roles.length} roadmaps` }}
          </span>
        </li>
      </ul>
    </div>
  </section>
</template>
