// k6 load test for XCRS (launch plan step 3). One scenario per run, at one steady rate: `run.sh` steps the rate
// up until a threshold fails, so each load level gets its own summary. The LLM is meant to be off on the target
// (XCRS_LLM_ENABLED=false) and rate limits disabled (XCRS_RATE_LIMITS_ENABLED=false); see loadtest/README.md.
//
//   k6 run -e BASE_URL=http://host:port -e SCENARIO=browse|api|recommend|match -e RATE=10 xcrs.js
import http from 'k6/http';
import { check } from 'k6';

const BASE = __ENV.BASE_URL || 'http://127.0.0.1:8080';
const SCENARIO = __ENV.SCENARIO || 'browse';
const RATE = Number(__ENV.RATE || 5); // iterations per second
const DURATION = __ENV.DURATION || '45s';

export const options = {
  scenarios: {
    [SCENARIO]: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1s',
      duration: DURATION,
      preAllocatedVUs: Math.min(20 + RATE * 2, 300),
      maxVUs: 600,
      exec: SCENARIO,
    },
  },
  // Safety stop: abort as soon as the target clearly can't keep up.
  thresholds: {
    http_req_failed: [{ threshold: 'rate<0.05', abortOnFail: true, delayAbortEval: '15s' }],
    http_req_duration: [{ threshold: 'p(95)<5000', abortOnFail: true, delayAbortEval: '15s' }],
  },
  summaryTrendStats: ['avg', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

const JSON_HEADERS = { headers: { 'Content-Type': 'application/json' } };
const SEARCH_TERMS = 'py jav kub doc sql rea aws ter git lin dat mac sec api tes clo net vue go rus'.split(' ');
const CATEGORIES = ['liked', 'liked', 'curious', 'curious', 'neutral', 'disliked'];
// Typed phrases, combined at random so most miss the phrase cache and run the embedding model.
const VERBS = 'building testing deploying designing debugging automating securing monitoring scaling teaching'.split(' ');
const OBJECTS = 'web apps data pipelines mobile games cloud servers dashboards databases models networks apis scripts'.split(' ');
const EXTRAS = 'for fun at work with friends in python on linux at scale for clients in teams quickly carefully'.split(' ');

const pick = (list) => list[Math.floor(Math.random() * list.length)];

export function setup() {
  const res = http.get(`${BASE}/api/v2/skills/groups`);
  check(res, { 'groups 200': (r) => r.status === 200 });
  const skills = res.json('groups').flatMap((g) => g.skills.map((s) => s.id));
  return { skills: [...new Set(skills)] };
}

// A visitor opening the board: the server-rendered page, a skill search, the quick-start groups.
export function browse() {
  check(http.get(`${BASE}/`), { 'home 200': (r) => r.status === 200 });
  check(http.get(`${BASE}/api/v2/skills?q=${pick(SEARCH_TERMS)}`), { 'search 200': (r) => r.status === 200 });
  check(http.get(`${BASE}/api/v2/skills/groups`), { 'groups 200': (r) => r.status === 200 });
}

// The API alone (no server-rendered page): a skill search and the quick-start groups. Used to reproduce
// API-only behaviour, e.g. against a local container.
export function api() {
  check(http.get(`${BASE}/api/v2/skills?q=${pick(SEARCH_TERMS)}`), { 'search 200': (r) => r.status === 200 });
  check(http.get(`${BASE}/api/v2/skills/groups`), { 'groups 200': (r) => r.status === 200 });
}

// A board of 4–8 catalog skills sent for recommendations (scoring + one database write).
export function recommend(data) {
  const n = 4 + Math.floor(Math.random() * 5);
  const chips = [];
  const used = new Set();
  while (chips.length < n) {
    const skill = pick(data.skills);
    if (used.has(skill)) continue;
    used.add(skill);
    chips.push({ category: pick(CATEGORIES), skill, proficiency: 1 + Math.floor(Math.random() * 4) });
  }
  const res = http.post(`${BASE}/api/v2/recommendations`, JSON.stringify({ chips }), JSON_HEADERS);
  check(res, { 'recommend 200': (r) => r.status === 200, 'has roles': (r) => (r.json('roles') || []).length > 0 });
}

// Two typed phrases matched by lookup and similarity (no LLM on the target).
export function match() {
  const phrases = [0, 1].map(() => `${pick(VERBS)} ${pick(OBJECTS)} ${pick(EXTRAS)}`);
  const res = http.post(`${BASE}/api/v2/skills/match`, JSON.stringify({ phrases }), JSON_HEADERS);
  check(res, { 'match 200': (r) => r.status === 200 });
}
