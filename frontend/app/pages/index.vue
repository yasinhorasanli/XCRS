<script setup lang="ts">
import { CATEGORIES } from '~/composables/categories'
import { EXAMPLES } from '~/composables/useBoardV2'
import { EXPERIENCE_OPTIONS } from '~/composables/useXcrsApiV2'

const { total, pending, clear, fillExample, asInput, experience, attract, startAttract, stopAttract, attractStep } = useBoardV2()
// Attract mode: the selected box follows the headline's flips until the visitor does anything on the board.
const turns = useState('headline-turns', () => 0)
watch(turns, attractStep)
watch(total, (n) => n && stopAttract())
const api = useXcrsApiV2()
const toast = useToast()
const submitting = ref(false)
const { restore, restoreLocal } = useSavedBoard() // signed in: the saved board comes back (ADR-0043)
onMounted(async () => {
  await restore()
  restoreLocal() // a reload (or a visit without an account) keeps the board this browser has
  startAttract()
})
onBeforeUnmount(() => (attract.value = false))

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
  <div class="page-width pb-16 pt-8 sm:pt-12">
    <!-- Full width, so the text lines up with the boxes below. -->
    <section>
      <h1 class="text-3xl font-bold tracking-tight text-highlighted sm:text-4xl">Find the software career that fits you</h1>
      <p class="mt-3 text-toned hyphens-auto sm:text-justify">
        Put the skills you know, and the ones you want to learn, into the four boxes. XCRS compares them with career
        roadmaps and recommends roles, your level in each, and what to learn next.
      </p>
    </section>

    <div class="mt-8">
      <div>
        <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <V2BucketV2 v-for="c in CATEGORIES" :key="c" :category="c" />
        </div>
        <!-- Quick start sits with the board it fills, under the boxes. -->
        <V2SuggestionsV2 class="mt-4" />
        <div class="mt-4 flex flex-wrap items-center gap-2 rounded-2xl bg-default px-3 py-2.5 shadow-xs ring-1 ring-default">
          <span class="mr-1 text-sm font-medium">Years in software <span class="font-normal text-muted">(optional)</span></span>
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
          <UDropdownMenu :items="[EXAMPLES.map((e) => ({ label: e.label, onSelect: () => fillExample(e.id) }))]">
            <UButton color="neutral" variant="soft" icon="i-heroicons-sparkles" trailing-icon="i-heroicons-chevron-down-20-solid" label="Try an example" />
          </UDropdownMenu>
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
          Tip: tap the dots on a skill to say how well you know it; it sharpens the level estimate. Your own words work
          too, like “building REST APIs with Django”.
        </p>
      </div>
    </div>
  </div>
</template>
