<script setup lang="ts">
import type { Category } from '~/types/api'

const props = defineProps<{
  label: string
  category?: Category // set when the chip sits in a bucket
  placed?: Category // for suggestions: the bucket it is already in, if any
  hint?: string // small text under the label, e.g. "because you added Docker"
}>()
const emit = defineEmits<{ remove: []; pick: [] }>()

const meta = computed(() => (props.category ? CATEGORY_META[props.category] : undefined))
const placedMeta = computed(() => (props.placed ? CATEGORY_META[props.placed] : undefined))

function onDragStart(event: DragEvent) {
  if (!event.dataTransfer) return
  event.dataTransfer.effectAllowed = 'move'
  event.dataTransfer.setData(DRAG_TYPE, JSON.stringify({ label: props.label, from: props.category }))
  event.dataTransfer.setData('text/plain', props.label)
}
</script>

<template>
  <span
    v-if="category"
    draggable="true"
    class="group inline-flex max-w-full cursor-grab items-center gap-1 rounded-full py-1 pl-3 pr-1 text-sm ring-1 ring-inset transition active:cursor-grabbing"
    :class="meta!.chip"
    @dragstart="onDragStart"
  >
    <span class="truncate">{{ label }}</span>
    <button
      type="button"
      class="grid h-5 w-5 shrink-0 place-items-center rounded-full opacity-60 transition hover:bg-black/10 hover:opacity-100 focus-visible:opacity-100"
      :aria-label="`Remove ${label}`"
      @click="emit('remove')"
    >
      <UIcon name="i-heroicons-x-mark-20-solid" class="h-3.5 w-3.5" />
    </button>
  </span>

  <button
    v-else
    type="button"
    draggable="true"
    class="inline-flex max-w-full cursor-grab flex-col items-start rounded-lg bg-white px-2.5 py-1 text-left text-sm shadow-sm ring-1 ring-inset ring-slate-200 transition hover:-translate-y-px hover:ring-indigo-300 hover:shadow active:cursor-grabbing"
    :class="placed ? 'opacity-60' : ''"
    :title="placed ? `Already in “${placedMeta!.title}”. Click to move it to the selected box.` : 'Click to add, or drag into a box'"
    @dragstart="onDragStart"
    @click="emit('pick')"
  >
    <span class="flex max-w-full items-center gap-1.5">
      <span v-if="placed" class="h-1.5 w-1.5 shrink-0 rounded-full" :class="placedMeta!.dot" />
      <span class="truncate">{{ label }}</span>
    </span>
    <span v-if="hint" class="max-w-full truncate text-[11px] leading-4 text-slate-400">{{ hint }}</span>
  </button>
</template>
