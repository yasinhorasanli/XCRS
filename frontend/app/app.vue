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
      <footer class="page-width grid gap-1.5 py-6 text-center text-xs text-dimmed print:hidden">
        <!-- Where XCRS comes from (2026-10-12): the thesis, the ISD 2025 paper and the archived first version. -->
        <p class="text-balance">
          XCRS began as a master's thesis at Boğaziçi University and was published at ISD 2025
          (<a href="https://doi.org/10.62036/ISD.2025.133" class="underline hover:text-toned" target="_blank" rel="noopener">paper</a>).
          Today's app is a rebuild shaped by its user study and expert evaluation
          (<a href="https://doi.org/10.5281/zenodo.14291086" class="underline hover:text-toned" target="_blank" rel="noopener">replication package</a>).
        </p>
        <p>
          Data: <NuxtLink to="/privacy#data-sources" class="underline hover:text-toned">O*NET 31.0 (CC BY 4.0)</NuxtLink>
          ·
          <NuxtLink to="/privacy" class="underline hover:text-toned">Privacy</NuxtLink>
        </p>
      </footer>
      <!-- On paper (the results PDF) links don't work: the attribution in full. -->
      <p class="hidden px-4 pt-6 text-center text-[10px] text-dimmed print:block">
        Includes information from the O*NET 31.0 Database by USDOL/ETA, used under CC BY 4.0, modified by XCRS; USDOL/ETA
        has not approved or endorsed it.
      </p>
    </div>
  </UApp>
</template>
