import type { Category, Experience, SkillRef } from '~/types/apiV2'

/**
 * The signed-in user's saved board (ADR-0043). The board page reloads it when it opens empty (`useSavedBoard`);
 * every change to the board, on any page (the board, or skills added from a result), is saved about a second
 * later (`useBoardAutosave`, set up once in app.vue).
 *
 * Every board, signed in or not, also has a copy in this browser's localStorage (2026-10-12), so a reload doesn't
 * lose it; it stays on the device and goes nowhere until the visitor asks for recommendations. Signing out clears
 * it (`forgetLocalBoard`).
 */

const SAVE_DELAY_MS = 1000
const LOCAL_KEY = 'xcrs-board'

type LocalBoard = {
  chips: { label: string; category: Category; skill?: string; proficiency?: number; skills?: SkillRef[]; method?: string }[]
  experience: Experience | null
}

/** Removes the browser's copy of the board (signing out, on a shared computer). */
export function forgetLocalBoard() {
  try {
    localStorage.removeItem(LOCAL_KEY)
  } catch {
    // storage blocked: nothing was kept
  }
}
let timer: ReturnType<typeof setTimeout> | undefined
let saveNow: (() => Promise<void>) | undefined
let saving: Promise<void> | undefined

export function useSavedBoard() {
  const { chips, experience, add } = useBoardV2()
  const { loggedIn } = useUserSession()
  const api = useXcrsApiV2()

  async function restore() {
    if (!loggedIn.value || chips.value.length) return
    try {
      const { board } = await api.board()
      // Typed chips are matched in one request (cached phrases answer at once), not one request per chip,
      // which a long board would run into the rate limit with (ADR-0035).
      const phrases = board?.chips.flatMap((c) => (c.skill || !c.text ? [] : [c.text])) ?? []
      const matches = phrases.length ? (await api.match(phrases.slice(0, 30)).catch(() => null))?.matches : undefined
      if (board && !chips.value.length) {
        for (const c of board.chips) {
          const label = c.name ?? c.text ?? c.skill ?? ''
          const matched = c.skill ? undefined : matches?.find((m) => m.phrase === c.text)
          add(label, c.category, c.skill ? { id: c.skill, name: label } : undefined, c.proficiency ?? undefined, matched)
        }
        experience.value = board.experience ?? null
      }
    } catch {
      // the board just starts empty
    }
  }

  /** The board from this browser, when the page opens empty (after the account's board, which wins). */
  function restoreLocal() {
    if (chips.value.length) return
    let saved: LocalBoard | null = null
    try {
      saved = JSON.parse(localStorage.getItem(LOCAL_KEY) ?? 'null')
    } catch {
      return
    }
    if (!saved?.chips?.length) return
    for (const c of saved.chips) {
      const matched = !c.skill && c.skills ? { skills: c.skills, method: c.method } : undefined
      add(c.label, c.category, c.skill ? { id: c.skill, name: c.label } : undefined, c.proficiency, matched)
    }
    experience.value = saved.experience ?? null
  }

  return { restore, restoreLocal }
}

/** Saves the board shortly after each change while someone is signed in. Call once, from app.vue. */
export function useBoardAutosave() {
  const { chips, experience, asInput } = useBoardV2()
  const { loggedIn } = useUserSession()
  const api = useXcrsApiV2()

  saveNow = async () => {
    clearTimeout(timer)
    timer = undefined
    saving = api
      .saveBoard(asInput(), experience.value)
      .then(() => undefined)
      .catch(() => undefined)
    await saving
  }

  // This browser's copy, for everyone: a reload brings the board back.
  watch(
    [chips, experience],
    () => {
      try {
        if (!chips.value.length && !experience.value) return localStorage.removeItem(LOCAL_KEY)
        const board: LocalBoard = {
          chips: chips.value.map((c) => ({
            label: c.label,
            category: c.category,
            ...(c.skill ? { skill: c.skill } : {}),
            ...(c.proficiency ? { proficiency: c.proficiency } : {}),
            // A typed chip keeps what it was matched to, so restoring it needs no new request.
            ...(!c.skill && c.match.status === 'done' ? { skills: c.match.skills, method: c.match.method } : {}),
          })),
          experience: experience.value,
        }
        localStorage.setItem(LOCAL_KEY, JSON.stringify(board))
      } catch {
        // storage blocked or full: the board just isn't kept across a reload
      }
    },
    { deep: true },
  )

  watch(
    [chips, experience],
    () => {
      if (!loggedIn.value) return
      clearTimeout(timer)
      timer = setTimeout(() => saveNow?.(), SAVE_DELAY_MS)
    },
    { deep: true },
  )

  // Leaving the page (reload, closing the tab, typing another address) before the delay is over: send the save
  // anyway. A keepalive request outlives the page; the session cookie goes with it (same site).
  window.addEventListener('pagehide', () => {
    if (!timer) return
    clearTimeout(timer)
    timer = undefined
    const body = JSON.stringify({ chips: asInput(), ...(experience.value ? { experience: experience.value } : {}) })
    fetch('/api/v2/me/board', { method: 'PUT', keepalive: true, headers: { 'content-type': 'application/json' }, body }).catch(
      () => undefined,
    )
  })
}

/** Writes a change that is still waiting to be saved, so a page that reads the saved board sees it. */
export async function flushSavedBoard() {
  if (timer) await saveNow?.()
  else await saving
}
