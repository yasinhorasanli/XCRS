<script setup lang="ts">
const route = useRoute()
const { loadProviders } = useAccount()
await callOnce('account-providers', loadProviders)
// Signed in, the board is saved after every change, wherever it happens (board page, results page).
if (import.meta.client) useBoardAutosave()
</script>

<template>
  <UApp :toaster="{ position: 'top-center' }">
    <div class="flex min-h-screen flex-col">
      <AppHeader class="print:hidden" />
      <!-- Home only: the wordmark and headline scroll away while the bar above stays. -->
      <HomeHero v-if="route.path === '/'" class="print:hidden" />
      <main class="flex-1">
        <NuxtPage />
      </main>
      <!-- The O*NET attribution (CC BY 4.0) is given in full on the privacy page; the link here carries it. -->
      <footer class="py-6 text-center text-xs text-dimmed print:hidden">
        Data: <NuxtLink to="/privacy#data-sources" class="underline hover:text-toned">O*NET 31.0 (CC BY 4.0)</NuxtLink>
        ·
        <NuxtLink to="/privacy" class="underline hover:text-toned">Privacy</NuxtLink>
      </footer>
      <!-- On paper (the results PDF) links don't work: the attribution in full. -->
      <p class="hidden px-4 pt-6 text-center text-[10px] text-dimmed print:block">
        Includes information from the O*NET 31.0 Database by USDOL/ETA, used under CC BY 4.0, modified by XCRS; USDOL/ETA
        has not approved or endorsed it.
      </p>
    </div>
  </UApp>
</template>
