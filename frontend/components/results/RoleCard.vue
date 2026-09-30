<script setup lang="ts">
import type { Role } from '~/types/api'

const props = defineProps<{
  role: Role
  index: number
  relativeFit: number // 0–1, compared with the top role
  requestId: string
  state: 'normal' | 'highlighted' | 'dimmed'
}>()

const color = computed(() => roleColor(props.index))
const fitLabel = computed(() => (props.index === 0 ? 'Best match' : props.relativeFit >= 0.75 ? 'Strong match' : 'Worth a look'))
</script>

<template>
  <article
    class="relative rounded-2xl bg-white p-5 shadow-sm ring-1 transition duration-200"
    :class="[
      state === 'highlighted' ? `ring-2 ${color.ring} shadow-md` : 'ring-slate-200',
      state === 'dimmed' ? 'opacity-40' : '',
    ]"
  >
    <header class="flex items-start gap-3">
      <span class="grid h-9 w-9 shrink-0 place-items-center rounded-xl text-sm font-bold" :class="color.badge">
        {{ index + 1 }}
      </span>
      <div class="min-w-0 flex-1">
        <h2 class="text-lg font-semibold leading-tight text-slate-900">{{ role.role }}</h2>
        <div class="mt-1.5 flex items-center gap-2">
          <div
            class="h-1.5 w-24 overflow-hidden rounded-full bg-slate-100"
            role="meter"
            :aria-valuenow="Math.round(relativeFit * 100)"
            aria-valuemin="0"
            aria-valuemax="100"
            :aria-label="`Fit compared with your best match`"
          >
            <div class="h-full rounded-full" :class="color.bar" :style="{ width: `${Math.max(8, relativeFit * 100)}%` }" />
          </div>
          <span class="text-xs font-medium" :class="color.text">{{ fitLabel }}</span>
        </div>
      </div>
      <ResultsFeedback :request-id="requestId" :role-id="role.role_id" :subject="role.role" />
    </header>

    <div class="mt-4">
      <h3 class="sr-only">Why this role</h3>
      <ResultsExplanation :text="role.explanation" :status="role.explanation_status" :lines="3" />
    </div>

    <div v-if="role.next_to_learn.length" class="mt-4">
      <h3 class="text-xs font-semibold uppercase tracking-wide text-slate-400">Next on this roadmap</h3>
      <div class="mt-1.5 flex flex-wrap gap-1">
        <span
          v-for="concept in role.next_to_learn.slice(0, 6)"
          :key="concept"
          class="rounded-md bg-slate-100 px-2 py-0.5 text-xs text-slate-600"
        >{{ concept }}</span>
      </div>
    </div>

    <!-- Small screens: no connector lines, so each role lists its own courses. -->
    <div class="mt-4 space-y-2 lg:hidden">
      <h3 class="text-xs font-semibold uppercase tracking-wide text-slate-400">Courses for this role</h3>
      <a
        v-for="c in role.courses"
        :key="c.course_id"
        :href="c.url"
        target="_blank"
        rel="noopener"
        class="block rounded-xl p-3 ring-1 ring-slate-200 hover:ring-slate-300"
      >
        <span class="flex items-start justify-between gap-2 text-sm font-medium text-slate-900">
          {{ c.title }}
          <UIcon name="i-heroicons-arrow-top-right-on-square" class="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
        </span>
        <span class="mt-1 block">
          <ResultsExplanation :text="c.explanation" :status="role.explanation_status" :lines="2" />
        </span>
      </a>
    </div>
  </article>
</template>
