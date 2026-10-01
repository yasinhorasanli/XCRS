<template>
  <UApp :toaster="{ position: 'top-center' }">
    <div class="flex min-h-screen flex-col">
      <AppHeader />
      <main class="flex-1">
        <NuxtPage />
      </main>
      <footer class="py-6 text-center text-xs text-slate-400">
        XCRS: roadmaps from <a class="underline hover:text-slate-600" href="https://roadmap.sh" target="_blank" rel="noopener">roadmap.sh</a>,
        courses from Udemy. Explanations are written by a local language model and can be imperfect.
      </footer>
    </div>
  </UApp>
</template>
