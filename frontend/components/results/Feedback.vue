<script setup lang="ts">
const props = defineProps<{ requestId: string; roleId?: number; courseId?: number; subject: string }>()

const api = useXcrsApi()
const vote = ref<1 | -1 | null>(null)
const sending = ref(false)

async function send(rating: 1 | -1) {
  if (vote.value || sending.value) return
  sending.value = true
  try {
    await api.feedback(props.requestId, { role_id: props.roleId, course_id: props.courseId, rating })
    vote.value = rating
  } catch {
    // feedback is optional; stay silent
  } finally {
    sending.value = false
  }
}
</script>

<template>
  <div class="flex items-center gap-1">
    <span v-if="vote" class="text-xs text-slate-400">Thanks!</span>
    <template v-else>
      <span class="mr-1 hidden text-xs text-slate-400 sm:inline">Useful?</span>
      <UButton
        size="2xs"
        color="gray"
        variant="ghost"
        icon="i-heroicons-hand-thumb-up"
        :aria-label="`${subject} is useful`"
        :disabled="sending"
        @click="send(1)"
      />
      <UButton
        size="2xs"
        color="gray"
        variant="ghost"
        icon="i-heroicons-hand-thumb-down"
        :aria-label="`${subject} is not useful`"
        :disabled="sending"
        @click="send(-1)"
      />
    </template>
  </div>
</template>
