<script setup lang="ts">
import { CATEGORIES, EXAMPLE_BOARD } from '~/composables/useBoard'
const { board, total, clear, fill } = useBoard()
const api = useXcrsApi()
const toast = useToast()
const submitting = ref(false)

async function submit() {
  submitting.value = true
  try {
    const result = await api.recommend(board.value)
    await navigateTo(`/classic/results/${result.request_id}`)
  } catch {
    toast.add({
      title: 'Could not get recommendations',
      description: 'The server did not answer. Please try again in a moment.',
      color: 'error',
      icon: 'i-heroicons-exclamation-triangle',
    })
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <div class="mx-auto max-w-7xl px-4 pb-16 pt-8 sm:pt-12">
    <section class="max-w-3xl">
      <p class="text-sm font-medium text-indigo-600">Career roles and courses, with the reasons</p>
      <h1 class="mt-2 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
        Map what you know. See which role fits, and the courses that get you there.
      </h1>
      <p class="mt-3 text-slate-600">
        Sort the things you've studied into how you felt about them, add what you're curious about, and XCRS matches
        them against ten career roadmaps. Every recommendation comes with an explanation built from your own input.
      </p>
      <ol class="mt-5 flex flex-wrap gap-x-6 gap-y-2 text-sm text-slate-600">
        <li class="flex items-center gap-2"><span class="grid h-5 w-5 place-items-center rounded-full bg-indigo-600 text-[11px] font-semibold text-white">1</span>Drag or type skills into the boxes</li>
        <li class="flex items-center gap-2"><span class="grid h-5 w-5 place-items-center rounded-full bg-indigo-600 text-[11px] font-semibold text-white">2</span>Get your top roles and courses</li>
        <li class="flex items-center gap-2"><span class="grid h-5 w-5 place-items-center rounded-full bg-indigo-600 text-[11px] font-semibold text-white">3</span>Read why each one was picked</li>
      </ol>
    </section>

    <div class="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_23rem]">
      <div>
        <div class="grid gap-4 sm:grid-cols-2">
          <BoardSkillBucket v-for="c in CATEGORIES" :key="c" :category="c" />
        </div>

        <div class="mt-5 flex flex-wrap items-center gap-2">
          <UButton color="neutral" variant="soft" icon="i-heroicons-sparkles" label="Try an example" @click="fill(EXAMPLE_BOARD)" />
          <UButton v-if="total" color="neutral" variant="ghost" icon="i-heroicons-trash" label="Clear all" @click="clear()" />
          <div class="ml-auto flex items-center gap-3">
            <span class="hidden text-sm text-slate-500 sm:inline">
              {{ total ? `${total} ${total === 1 ? 'skill' : 'skills'} added` : 'Add at least one skill' }}
            </span>
            <UButton
              size="lg"
              :disabled="!total"
              :loading="submitting"
              trailing-icon="i-heroicons-arrow-right-20-solid"
              label="Get recommendations"
              @click="submit"
            />
          </div>
        </div>
        <p class="mt-3 text-xs text-slate-500">
          Tip: a few specific skills work better than many vague ones. Your own words are fine; they don't have to match a suggestion.
        </p>
      </div>

      <div class="lg:sticky lg:top-20 lg:self-start">
        <BoardSuggestionPanel />
      </div>
    </div>
  </div>
</template>
