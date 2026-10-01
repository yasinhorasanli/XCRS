<script setup lang="ts">
const route = useRoute()
const isV2 = computed(() => route.path.startsWith('/v2'))
</script>

<template>
  <UApp :toaster="{ position: 'top-center' }">
    <div class="flex min-h-screen flex-col">
      <AppHeader />
      <main class="flex-1">
        <NuxtPage />
      </main>
      <footer v-if="isV2" class="py-6 text-center text-xs text-slate-400">
        XCRS new engine (beta). Includes information from the
        <a class="underline hover:text-slate-600" href="https://www.onetcenter.org/database.html" target="_blank" rel="noopener">O*NET 31.0 Database</a>
        by USDOL/ETA, used under CC BY 4.0, modified by XCRS; USDOL/ETA has not approved or endorsed it.
        Level estimates come from your input and can be off.
      </footer>
      <footer v-else class="py-6 text-center text-xs text-slate-400">
        XCRS: roadmaps from <a class="underline hover:text-slate-600" href="https://roadmap.sh" target="_blank" rel="noopener">roadmap.sh</a>,
        courses from Udemy. Explanations are written by a local language model and can be imperfect.
      </footer>
    </div>
  </UApp>
</template>
