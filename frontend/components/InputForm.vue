<script setup lang="ts">
import { object, string, type InferType } from 'yup'
import type { FormSubmitEvent } from '#ui/types'
import type { RecommendationResult } from '~/models/result';

const schema = object({
  took_and_liked: string()
    .required('Required'),
  took_and_neutral: string()
    .required('Required'),
  took_and_disliked: string()
    .required('Required'),
  curious: string()
    //.min(4, 'Must be at least 4 characters')
    .required('Required')
})

type Schema = InferType<typeof schema>

const state = reactive({
  took_and_liked: undefined,
  took_and_neutral: undefined,
  took_and_disliked: undefined,
  curious: undefined
})

const emits = defineEmits<{
  (e: 'submited', result: RecommendationResult): void
}>();


async function onSubmit(event: FormSubmitEvent<Schema>) {
  // Do something with event.data

  try {
    const response = await $fetch<RecommendationResult>('/api/recommend', {
      method: 'POST',
      body: event.data
    });

    console.log('userData =', event.data.curious)

    console.log('userData =', response)


    emits('submited', response);
  } catch {
    // hata göster
  }


}
</script>

<template>
  <UForm :schema="schema" :state="state" class="space-y-4" @submit="onSubmit">

    <UFormGroup label="Took and Liked" name="took_and_liked">
      <UTextarea autoresize :maxrows="5" placeholder="Courses or topics you took and liked"
        v-model="state.took_and_liked" />
    </UFormGroup>

    <UFormGroup label="Took and Neutral" name="took_and_neutral">
      <UTextarea autoresize :maxrows="5" placeholder="Courses or topics you took and neither you liked nor disliked..."
        v-model="state.took_and_neutral" />
    </UFormGroup>

    <UFormGroup label="Took and Disliked" name="took_and_disliked">
      <UTextarea autoresize :maxrows="5" placeholder="Courses or topics you took and disliked"
        v-model="state.took_and_disliked" />
    </UFormGroup>
    <UFormGroup label="Curious" name="curious">
      <UTextarea autoresize :maxrows="5" placeholder="Courses or topics you are curious about"
        v-model="state.curious" />
    </UFormGroup>

    <UButton type="submit">
      Submit
    </UButton>
  </UForm>
</template>