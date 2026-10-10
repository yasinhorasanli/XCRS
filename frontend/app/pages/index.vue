<script setup lang="ts">
import { CATEGORIES } from '~/composables/categories'
import { EXPERIENCE_OPTIONS } from '~/composables/useXcrsApiV2'

const { total, pending, clear, fillExample, asInput, experience } = useBoardV2()
const api = useXcrsApiV2()
const toast = useToast()
const submitting = ref(false)
const { restore } = useSavedBoard() // signed in: the saved board comes back (ADR-0043)
onMounted(restore)

useHead({ title: 'XCRS · Which software career fits you?' })

async function submit() {
  submitting.value = true
  try {
    const result = await api.recommend(asInput(), experience.value)
    await navigateTo(`/results/${result.id}`)
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
      <p class="flex items-center gap-2 text-sm font-medium text-indigo-600 dark:text-indigo-300">
        Career roles, your level, and what to learn next
      </p>
      <h1 class="mt-2 text-3xl font-bold tracking-tight text-highlighted sm:text-4xl">Which of 30 software careers fits you, and where would you start?</h1>
      <p class="mt-3 text-toned">
        Add skills you know (and how well), what you didn't enjoy, and what you're curious about. XCRS compares them with
        the roadmaps of 30 roles, from entry to staff level, and shows the skills that would take you to the next level.
      </p>
    </section>

    <div class="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_23rem]">
      <div>
        <div class="grid gap-4 sm:grid-cols-2">
          <V2BucketV2 v-for="c in CATEGORIES" :key="c" :category="c" />
        </div>
        <div class="mt-5 flex flex-wrap items-center gap-2 rounded-2xl bg-default px-3 py-2.5 shadow-xs ring-1 ring-default">
          <span class="mr-1 text-sm font-medium">Years in software</span>
          <span class="mr-2 text-xs text-muted">(optional; improves the level estimate)</span>
          <div class="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Years in software">
            <button
              v-for="o in EXPERIENCE_OPTIONS"
              :key="o.value"
              type="button"
              role="radio"
              :aria-checked="experience === o.value"
              class="rounded-full px-3 py-1 text-sm ring-1 ring-inset transition"
              :class="experience === o.value ? 'bg-indigo-600 dark:bg-indigo-700 text-white ring-indigo-600 dark:ring-indigo-500' : 'bg-default text-default ring-default hover:bg-muted'"
              @click="experience = experience === o.value ? null : o.value"
            >
              {{ o.label }}
            </button>
          </div>
        </div>
        <div class="mt-4 flex flex-wrap items-center gap-2">
          <UButton color="neutral" variant="soft" icon="i-heroicons-sparkles" label="Try an example" @click="fillExample()" />
          <CvImport />
          <UButton v-if="total" color="neutral" variant="ghost" icon="i-heroicons-trash" label="Clear all" @click="clear()" />
          <div class="ml-auto flex items-center gap-3">
            <span class="hidden text-sm text-muted sm:inline">
              <template v-if="pending">Reading {{ pending }} of your entries…</template>
              <template v-else>{{ total ? `${total} ${total === 1 ? 'skill' : 'skills'} added` : 'Add at least one skill' }}</template>
            </span>
            <UButton size="lg" :disabled="!total" :loading="submitting" trailing-icon="i-heroicons-arrow-right-20-solid" label="Show my roles" @click="submit" />
          </div>
        </div>
        <p class="mt-3 text-xs text-muted">
          Tip: rate how well you know a skill with the dots; it makes the level estimate much better. Your own words work
          too ("building REST APIs with Django"); we show which skills we read them as.
        </p>
      </div>
      <div class="lg:sticky lg:top-20 lg:self-start">
        <V2SuggestionsV2 />
      </div>
    </div>
  </div>
</template>
