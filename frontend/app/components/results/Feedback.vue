<script setup lang="ts">
const props = defineProps<{ requestId: string; roleId?: number; courseId?: number; subject: string }>()

const api = useXcrsApi()
const vote = ref<1 | -1 | null>(null)
const sending = ref(false)

// Votes are anonymous, so the browser remembers them: a reload shows "Thanks!" instead of
// inviting a second vote on the same role or course. Storage can be unavailable (private mode).
const storageKey = computed(
  () => `xcrs-feedback:${props.requestId}:${props.roleId ?? '-'}:${props.courseId ?? '-'}`,
)
onMounted(() => {
  try {
    const saved = localStorage.getItem(storageKey.value)
    if (saved === '1' || saved === '-1') vote.value = Number(saved) as 1 | -1
  } catch {
    // no storage: voting still works, it just isn't remembered
  }
})

async function send(rating: 1 | -1) {
  if (vote.value || sending.value) return
  sending.value = true
  try {
    await api.feedback(props.requestId, { role_id: props.roleId, course_id: props.courseId, rating })
    vote.value = rating
    try {
      localStorage.setItem(storageKey.value, String(rating))
    } catch {
      // see above
    }
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
        size="xs"
        color="neutral"
        variant="ghost"
        icon="i-heroicons-hand-thumb-up"
        :aria-label="`${subject} is useful`"
        :disabled="sending"
        @click="send(1)"
      />
      <UButton
        size="xs"
        color="neutral"
        variant="ghost"
        icon="i-heroicons-hand-thumb-down"
        :aria-label="`${subject} is not useful`"
        :disabled="sending"
        @click="send(-1)"
      />
    </template>
  </div>
</template>
