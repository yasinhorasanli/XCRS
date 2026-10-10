<script setup lang="ts">
// The new engine moved from /v2 to / (ADR-0037); keep old result links working.
const route = useRoute()
await navigateTo(`/results/${route.params.id}`, { redirectCode: 301 })
</script>

<template>
  <div />
</template>
