<script setup lang="ts">
useHead({ title: 'Privacy · XCRS' })
</script>

<template>
  <div class="page-width pb-16 pt-8 text-default">
    <!-- Same width as every page (the header stays put), text included. -->
    <div>
      <h1 class="text-2xl font-bold tracking-tight text-highlighted">Privacy notice</h1>
      <p class="mt-2 text-sm text-muted">Last updated 12 October 2026</p>

      <!-- Justified from tablet width up; on phones, justified lines leave wide gaps between words. -->
      <div class="mt-6 grid gap-6 leading-relaxed hyphens-auto sm:text-justify">
        <section>
          <h2 class="text-lg font-semibold text-highlighted">In short</h2>
          <p class="mt-2">
            XCRS is a free, non-commercial project run by one person. It works without an account. If you sign in, we keep
            what is needed to show you your results again, and nothing more. There are no ads, no analytics and no tracking
            cookies, and nothing is sold or shared.
          </p>
        </section>

        <section>
          <h2 class="text-lg font-semibold text-highlighted">Without an account</h2>
          <p class="mt-2">
            When you ask for recommendations, we store the skills on your board, your optional years of experience, the
            results, any thumbs or comments you give, and when this happened. Nothing links this to you. Anyone with a
            result's link can open it, so share links with care. To limit abuse, your IP address is held briefly in memory
            for rate limiting; it is not stored in the database.
          </p>
        </section>

        <section>
          <h2 class="text-lg font-semibold text-highlighted">With an account</h2>
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
          <h2 class="text-lg font-semibold text-highlighted">Importing your CV or LinkedIn PDF</h2>
          <p class="mt-2">
            If you're signed in, you can let XCRS read a CV or LinkedIn's "Save to PDF" of your profile, or text you paste,
            to suggest skills for your board. The file is read in memory on our own server, and our own language model
            (running on that server, not a third-party service) picks out skills and the dates of your jobs. Nothing from
            the file is stored: not the file, its text or what the model found. The suggestions are kept in memory for at
            most 15 minutes, until you see them, and our logs record only the size of the text, not its content. Only the
            skills you choose to add to your board are kept, like any board, and saved to your account. Hidden text and
            lines written as instructions to an AI are ignored, and we tell you when that happens.
          </p>
        </section>

        <section>
          <h2 class="text-lg font-semibold text-highlighted">Cookies</h2>
          <p class="mt-2">
            Only cookies the site needs: when you sign in, an encrypted session cookie (for 30 days) and, during the
            sign-in itself, short-lived cookies that protect it. No cookies are set if you don't sign in.
            Your colour-mode choice is stored only in your browser (localStorage) and is not sent to us.
          </p>
        </section>

        <section>
          <h2 class="text-lg font-semibold text-highlighted">Your rights</h2>
          <p class="mt-2">
            On your <NuxtLink to="/account" class="text-indigo-600 dark:text-indigo-300 underline">account page</NuxtLink> you can download
            everything we keep about you (as JSON), delete single results, sign out on all devices, and delete your account.
            Deleting the account deletes your board, your results and the feedback on them, at once. Backups of the
            database are kept for up to 14 days and then replaced, so deleted data leaves them within that time.
          </p>
          <p class="mt-2">
            For any other question about your data (access, correction, objection), open an issue or contact the maintainer
            through the <a href="https://github.com/yasinhorasanli/XCRS" class="text-indigo-600 dark:text-indigo-300 underline" target="_blank" rel="noopener">project's GitHub page</a>.
          </p>
        </section>

        <section>
          <h2 class="text-lg font-semibold text-highlighted">Where the data is</h2>
          <p class="mt-2">
            On servers the project runs itself. The explanations are written by a language model on those servers; your
            data isn't sent to an outside AI service. Visitors reach the site through Cloudflare, which handles the secure
            connection and passes requests on to our server.
          </p>
        </section>

        <section id="data-sources" class="scroll-mt-20">
          <h2 class="text-lg font-semibold text-highlighted">Data sources</h2>
          <p class="mt-2">
            XCRS includes information from the
            <a href="https://www.onetcenter.org/database.html" class="text-indigo-600 dark:text-indigo-300 underline" target="_blank" rel="noopener">O*NET 31.0 Database</a>
            by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA), used under the
            <a href="https://creativecommons.org/licenses/by/4.0/" class="text-indigo-600 dark:text-indigo-300 underline" target="_blank" rel="noopener">CC BY 4.0</a>
            license. XCRS has modified the information. USDOL/ETA has not approved, endorsed or tested these modifications.
          </p>
        </section>

        <section>
          <h2 class="text-lg font-semibold text-highlighted">The demo copy at aws.xcrs.app</h2>
          <p class="mt-2">
            A copy of XCRS runs on Amazon Web Services (AWS) in Frankfurt, behind Amazon CloudFront, to try out cloud
            hosting. It has no sign-in. If you use it, what "Without an account" describes is stored in its own database
            on AWS, separate from xcrs.app, and is deleted whenever the copy is taken down. Its explanations are written by
            the same language model on our own servers.
          </p>
        </section>
      </div>
    </div>
  </div>
</template>
