<script setup lang="ts">
// The new engine moved from /v2 to / (ADR-0037); keep old links working.
await navigateTo('/', { redirectCode: 301 })
</script>

<template>
  <div />
</template>
