<script setup lang="ts">
// Each flip of the headline's word moves the board's selected box while attract mode is on (useBoardV2).
const turns = useState('headline-turns', () => 0)
</script>

<!-- Home page top: the large wordmark and the rotating headline. -->
<template>
  <div class="page-width pt-8 pb-2 text-center sm:pt-12">
    <img
      src="/xcrs-logo.webp"
      alt="XCRS"
      width="766"
      height="240"
      class="mx-auto h-16 w-auto sm:h-24 lg:h-28"
      fetchpriority="high"
    >
    <RotatingHeadline @turn="turns++" class="mt-3 text-[clamp(1.75rem,4.2vw,3.5rem)] leading-[1.15] font-extrabold tracking-tight sm:mt-5" />
  </div>
</template>
