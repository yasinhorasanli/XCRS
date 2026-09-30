<script setup lang="ts">
import type { ExplanationStatus } from '~/types/api'

withDefaults(defineProps<{ text: string | null; status: ExplanationStatus; lines?: number }>(), { lines: 3 })
</script>

<template>
  <p v-if="text" class="text-sm leading-relaxed text-slate-600">{{ text }}</p>
  <div v-else-if="status === 'pending'" aria-live="polite">
    <span class="sr-only">Writing the explanation…</span>
    <div class="space-y-1.5" aria-hidden="true">
      <div v-for="i in lines" :key="i" class="shimmer h-3 rounded" :style="{ width: i === lines ? '60%' : '100%' }" />
    </div>
  </div>
  <p v-else-if="status === 'failed'" class="text-sm italic text-slate-400">
    The explanation couldn't be written this time. The recommendation itself stands.
  </p>
</template>
