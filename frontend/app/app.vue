<template>
  <UApp :toaster="{ position: 'top-center' }">
    <div class="flex min-h-screen flex-col">
      <AppHeader />
      <main class="flex-1">
        <NuxtPage />
      </main>
      <footer class="py-6 text-center text-xs text-slate-400">
        XCRS. Includes information from the
        <a class="underline hover:text-slate-600" href="https://www.onetcenter.org/database.html" target="_blank" rel="noopener">O*NET 31.0 Database</a>
        by USDOL/ETA, used under CC BY 4.0, modified by XCRS; USDOL/ETA has not approved or endorsed it.
        Level estimates come from your input and can be off.
      </footer>
    </div>
  </UApp>
</template>
