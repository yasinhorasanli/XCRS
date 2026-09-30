<script setup lang="ts">
import type { CourseGroup } from '~/composables/useResultView'

const props = defineProps<{
  course: CourseGroup
  requestId: string
  state: 'normal' | 'highlighted' | 'dimmed'
  highlightedRole: number | null // role id being hovered, if any
}>()

// The roadmap concepts it covers, across all roles that picked it.
const concepts = computed(() => [...new Set(props.course.links.flatMap((l) => l.concepts))])
const shared = computed(() => props.course.links.length > 1)
</script>

<template>
  <article
    class="rounded-2xl bg-white p-4 shadow-sm ring-1 transition duration-200"
    :class="[
      state === 'highlighted' ? 'shadow-md ring-2 ring-slate-400' : 'ring-slate-200',
      state === 'dimmed' ? 'opacity-40' : '',
    ]"
  >
    <header class="flex items-start justify-between gap-3">
      <div class="min-w-0">
        <div class="mb-1 flex flex-wrap gap-1">
          <span
            v-for="l in course.links"
            :key="l.roleId"
            class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset"
            :class="roleColor(l.roleIndex).tag"
          >
            <span class="h-1.5 w-1.5 rounded-full" :class="roleColor(l.roleIndex).dot" />{{ l.role }}
          </span>
          <span v-if="shared" class="rounded-full bg-slate-900 px-2 py-0.5 text-[11px] font-medium text-white">
            Fits {{ course.links.length }} roles
          </span>
        </div>
        <a
          :href="course.url"
          target="_blank"
          rel="noopener"
          class="group inline-flex items-start gap-1 font-semibold leading-snug text-slate-900 hover:text-indigo-700"
        >
          {{ course.title }}
          <UIcon name="i-heroicons-arrow-top-right-on-square" class="mt-0.5 h-4 w-4 shrink-0 text-slate-400 group-hover:text-indigo-500" />
        </a>
      </div>
      <ResultsFeedback :request-id="requestId" :course-id="course.course_id" :subject="course.title" />
    </header>

    <div class="mt-3 space-y-2.5">
      <div v-for="l in course.links" :key="l.roleId" :class="shared && highlightedRole && highlightedRole !== l.roleId ? 'opacity-50' : ''">
        <p v-if="shared" class="mb-0.5 text-[11px] font-semibold uppercase tracking-wide" :class="roleColor(l.roleIndex).text">
          For {{ l.role }}
        </p>
        <ResultsExplanation :text="l.explanation" :status="l.status" :lines="2" />
      </div>
    </div>

    <div v-if="concepts.length" class="mt-3 flex flex-wrap items-center gap-1">
      <span class="mr-0.5 text-[11px] text-slate-400">Covers</span>
      <span v-for="c in concepts.slice(0, 5)" :key="c" class="rounded-md bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-600">{{ c }}</span>
      <span v-if="concepts.length > 5" class="text-[11px] text-slate-400">+{{ concepts.length - 5 }} more</span>
    </div>
  </article>
</template>
