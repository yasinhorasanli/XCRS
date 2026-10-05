<script setup lang="ts">
useHead({ title: 'Privacy · XCRS' })
</script>

<template>
  <div class="mx-auto max-w-3xl px-4 pb-16 pt-8 text-slate-700">
    <h1 class="text-2xl font-bold tracking-tight text-slate-900">Privacy notice</h1>
    <p class="mt-2 text-sm text-slate-500">Last updated 3 October 2026</p>

    <div class="mt-6 grid gap-6 leading-relaxed">
      <section>
        <h2 class="text-lg font-semibold text-slate-900">In short</h2>
        <p class="mt-2">
          XCRS is a free, non-commercial project run by one person. It works without an account. If you sign in, we keep
          what is needed to show you your results again, and nothing more. There are no ads, no analytics and no tracking
          cookies, and nothing is sold or shared.
        </p>
      </section>

      <section>
        <h2 class="text-lg font-semibold text-slate-900">Without an account</h2>
        <p class="mt-2">
          When you ask for recommendations, we store the skills on your board, your optional years of experience, the
          results, any thumbs or comments you give, and when this happened. Nothing links this to you. Anyone with a
          result's link can open it, so share links with care. To limit abuse, your IP address is held briefly in memory
          for rate limiting; it is not stored in the database.
        </p>
      </section>

      <section>
        <h2 class="text-lg font-semibold text-slate-900">With an account</h2>
        <p class="mt-2">You can sign in with GitHub, Google or LinkedIn. From them we keep:</p>
        <ul class="mt-2 list-disc space-y-1 pl-5">
          <li>your name and the provider's id for your account;</li>
          <li>your email address, only if the provider has verified it (accounts with the same verified email are joined into one);</li>
          <li>when you joined and last signed in.</li>
        </ul>
        <p class="mt-2">
          We don't keep passwords, profile pictures or the provider's access tokens. Your board and the results you make
          while signed in (or save to your account) are linked to your account.
        </p>
      </section>

      <section>
        <h2 class="text-lg font-semibold text-slate-900">Cookies</h2>
        <p class="mt-2">
          Only cookies the site needs: when you sign in, an encrypted session cookie (for 30 days) and, during the
          sign-in itself, short-lived cookies that protect it. No cookies are set if you don't sign in.
        </p>
      </section>

      <section>
        <h2 class="text-lg font-semibold text-slate-900">Your rights</h2>
        <p class="mt-2">
          On your <NuxtLink to="/account" class="text-indigo-600 underline">account page</NuxtLink> you can download
          everything we keep about you (as JSON), delete single results, sign out on all devices, and delete your account.
          Deleting the account deletes your board, your results and the feedback on them, at once. Backups of the
          database are kept for up to 14 days and then replaced, so deleted data leaves them within that time.
        </p>
        <p class="mt-2">
          For any other question about your data (access, correction, objection), open an issue or contact the maintainer
          through the <a href="https://github.com/yasinhorasanli/XCRS" class="text-indigo-600 underline" target="_blank" rel="noopener">project's GitHub page</a>.
        </p>
      </section>

      <section>
        <h2 class="text-lg font-semibold text-slate-900">Where the data is</h2>
        <p class="mt-2">
          On servers the project runs itself. The explanations are written by a language model on those servers; your
          data isn't sent to an outside AI service. Visitors reach the site through Tailscale Funnel, which relays the
          encrypted connection without reading it.
        </p>
      </section>
    </div>
  </div>
</template>
