<script setup lang="ts">
import { CATEGORY_META } from '~/composables/categories'
import type { Chip } from '~/composables/useBoardV2'
import { PROFICIENCY_NAMES } from '~/composables/useXcrsApiV2'

const props = defineProps<{ chip: Chip }>()
const emit = defineEmits<{ remove: []; rate: [level: number] }>()

const meta = computed(() => CATEGORY_META[props.chip.category])
// Curious means "not known yet": no proficiency to rate.
const rateable = computed(() => props.chip.category !== 'curious')
const matchedNames = computed(() => props.chip.match.skills.map((s) => s.name).join(', '))
const showMatch = computed(() => !props.chip.skill && props.chip.match.status !== 'pending')
</script>

<template>
  <span class="inline-flex max-w-full flex-col rounded-2xl py-1 pl-3 pr-1 text-sm ring-1 ring-inset" :class="meta.chip">
    <span class="flex max-w-full items-center gap-1.5">
      <span class="truncate">{{ chip.label }}</span>
      <UIcon v-if="chip.match.status === 'pending'" name="i-heroicons-arrow-path" class="h-3.5 w-3.5 shrink-0 animate-spin opacity-60" aria-label="Matching to skills" />
      <span v-if="rateable" class="ml-0.5 flex shrink-0 items-center gap-0.5" role="group" :aria-label="`How well you know ${chip.label}`">
        <button
          v-for="n in 4"
          :key="n"
          type="button"
          class="h-2.5 w-2.5 rounded-full ring-1 ring-current transition hover:scale-125"
          :class="(chip.proficiency ?? 0) >= n ? 'bg-current opacity-80' : 'opacity-40'"
          :title="`${PROFICIENCY_NAMES[n]}${chip.proficiency === n ? ' (click again to clear)' : ''}`"
          :aria-label="`${PROFICIENCY_NAMES[n]}`"
          :aria-pressed="chip.proficiency === n"
          @click="emit('rate', n)"
        />
      </span>
      <button
        type="button"
        class="grid h-5 w-5 shrink-0 place-items-center rounded-full opacity-60 transition hover:bg-black/10 hover:opacity-100 focus-visible:opacity-100"
        :aria-label="`Remove ${chip.label}`"
        @click="emit('remove')"
      >
        <UIcon name="i-heroicons-x-mark-20-solid" class="h-3.5 w-3.5" />
      </button>
    </span>
    <span v-if="showMatch" class="max-w-full truncate pr-2 text-[11px] leading-4 opacity-75">
      <template v-if="chip.match.skills.length">→ {{ matchedNames }}</template>
      <template v-else-if="chip.match.status === 'error'">will be matched when you submit</template>
      <template v-else><span class="text-amber-700">not a skill we know; try other words</span></template>
    </span>
  </span>
</template>
