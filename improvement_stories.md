# CrossScan: engineering improvement stories

Interview notes for CrossScan, an agentic multimodal RAG over 12 research papers (lung-cancer imaging, land-cover remote sensing, one AI-security paper). Every story has the same shape: **problem, why it happened, how it was fixed, evidence, a 30-second version**. Numbers come from real runs in this project; the honest limits are stated so you do not over-claim.

Reports behind the numbers: `reports/live_query_20260919_204144.md` (before), `reports/live_query_20260919_222248.md` (after), `reports/latency_compare_20260919_205533.md` (cold / warm / cached).

---

## Project at a glance

| | |
|---|---|
| Pipeline | LangGraph: input guardrail, answer cache, Agent 1 (query planner), retrieval (pgvector + Neo4j graph), quality check, reranker, answer generation, output guardrail |
| Data | 12 papers: 316 sections, 174 images in a Neo4j graph (65 methods, 16 metrics, 7 datasets) plus pgvector chunks |
| Models | Google Gemini only, in 3 tiers (fast, reasoning, evaluation), 2 models each, automatic fallback within a tier |
| App layer | Streamlit chat UI, FastAPI, login and roles, per-user quotas, conversation memory, feedback |
| Evaluation | Offline: hand-written golden set (36 questions, DVC-versioned), DeepEval and RAGAS, MLflow. Online: feedback, sampled LLM judge, health report |
| Tests | **565**: 529 unit + 33 integration (real Postgres) + 3 live (real Gemini and Neo4j). 7,189 test lines against 6,096 source lines (ratio 1.18) |

---

## Story 1: Cutting cost and latency of one question (the headline story)

**Situation.** First live question: *"What accuracy did the VGG16 and CNN models achieve for lung cancer CT scan classification, and which performed better?"* The answer was correct, but the run was slow and expensive for a single lookup.

**Measured (instrumented per stage and per model call):**

| | Before | After | Change |
|---|---|---|---|
| Model calls | 5 | **3** | -40% |
| Prompt tokens | 32,015 | **14,609** | **-54%** |
| Output tokens | 1,116 | 581 | -48% |
| Pipeline time | 86.9 s | **58.2 s** | **-33%** |
| Sub-questions | 2 overlapping | 1 | |
| Source pages | 7 | 5 | |

**Why it was expensive.** Reading the per-call table showed the cause. The planner split one question into two sub-questions ("what accuracy...", "which performed better...") that both needed the *same* results passage. Each sub-question ran its own retrieval, its own quality check, and its own answer. So the same ~7,800-token context was sent to a model **four times** (2 quality checks + 2 answers). Both were also labelled `complex`, so they used the stronger reasoning tier.

**Fixes (layered, so one failing still leaves the others):**
1. *Prompt (prevent it):* tell the planner to split only when parts need **different lookups**, and give this exact question as the example of one question.
2. *Code (catch it anyway):* after retrieval, merge sub-questions whose retrieved passages overlap by 80% or more, measured against the smaller set. Prompts are probabilistic, so a deterministic safety net matters.
3. *Removed doomed requests:* skip the prompt-cache request for prompts under 1,024 tokens (Google refuses them).
4. *Faster failover:* see Story 2.

```python
def merge_overlapping_sub_queries(sub_queries, threshold=0.8):
    ...
    smaller = min(len(mine), len(theirs))
    if smaller and len(mine & theirs) / smaller >= threshold:   # overlap vs the SMALLER set
        target = other
    ...
    target["sub_query"] = f"{target['sub_query'].rstrip()} Also: {sq['sub_query'].lstrip()}"
    if "complex" in (target.get("complexity"), sq.get("complexity")):
        target["complexity"] = "complex"                        # keep the stronger label
```

**A bug I found while doing it.** The second answer cache is *read* before retrieval and *written* after it. Merging changes the sub-question text, so the written key would differ from the read key and merged questions could never hit the cache. Fixed by saving the pre-merge key in the graph state (`combined_key`). Covered by a test.

**30-second version.** "I instrumented the pipeline per stage and per model call and found one question was sending the same 7.8k-token context to the model four times, because the planner split it into overlapping sub-questions. I fixed it in two layers, a prompt change and a deterministic merge of sub-questions with heavily overlapping retrieval. Prompt tokens dropped 54% and latency about a third on the same question."

**Honest limits.** One run, and LLM output varies. The planner did not split the question this time, so the merge safety net was **not exercised against a live model** (it has unit tests only). Latency is also dominated by Gemini's own response time and 503s. The sub-question was still labelled `complex` (see "What is not fixed").

---

## Story 2: "Server busy" is not a quota error (failover design)

**Situation.** Calls sometimes failed with "Gemini server busy". Natural assumption: free-tier limit hit. It was not: the free-tier limit had not been used that day.

**Why.** The retry code catches `ServerError`, meaning HTTP 5xx. Quota exhaustion is HTTP 429 and is classified separately (`quota`, 15-minute cooldown). The live log later confirmed the real code: `503 UNAVAILABLE`, an overload on Google's side. Newer models seem to have this more often; free-tier traffic tends to be served last under load (my inference, not proven).

**Problems with the old behaviour.**
- The message was generic, so the real error code was hidden.
- A busy model was retried three times (waits of 2 + 4 + 8 s) **even when a healthy fallback model was waiting**. In the first report one failing call cost 35 s.

**Fix.**
- Log the model and real status: `Gemini gemini-3.6-flash returned 503 UNAVAILABLE: retrying in 2s`.
- Retry budget depends on the situation: a model with a healthy fallback later in its tier gets **one** retry, then the system fails over. The last usable model keeps the full four attempts, since there is nowhere left to go.

```python
fallback_ready = any(
    _unavailable_reason(other["provider"], other["model"], client) is None
    for other in TIERS[tier][position + 1:]
)
config = {**PROVIDERS[provider], "max_retries": FAILOVER_RETRIES if fallback_ready else FULL_RETRIES}
```

**Evidence.** In the re-run the log showed one 2 s wait and then `tier 'fast' answered by fallback model gemini-3.5-flash-lite`. The next call in the same request went straight to the fallback with no retry, consistent with the 60 s cooldown on a model that just failed.

**30-second version.** "Transient overload and quota exhaustion look similar but need different reactions. I classify errors by HTTP status: 429 gets a long per-model cooldown, 5xx gets a short one. Retries are cheap only when nothing else can serve the request, so I cut retries to one when a healthy fallback exists, and log the real status so incidents are diagnosable."

---

## Story 3: Cold vs warm vs cached: testing an assumption

**Situation.** The idea: "most things are loaded, so running the same test again must be faster." I measured it in one process.

| Run | Wall time | Model calls |
|---|---|---|
| 1. Cold (first question in the process) | 57.7 s | 5 |
| 2. Warm (same question, answer cache off) | **65.8 s** | 5 |
| 3. Cached (answer cache on) | **0.17 s** | 0 |

**What actually happened.** Warm-up was real but small: retrieval fell from 17.0 s to 5.2 s (models already in memory, about 12 s saved). It was swamped by two other things: a 503 in the planner (10 s became 37 s) and non-determinism (a different plan meant 40k prompt tokens instead of 30k and a longer answer). The answer cache was the big win (0.17 s, zero calls, identical answer) but only for exact repeats.

**Lesson.** Measure before optimising. Model load is a one-time cost per process (about 45-50 s of imports plus about 12 s of model load), so it belongs at server start, not on the request path. Only the exact-repeat case is helped by the answer cache.

**30-second version.** "I assumed a second run would be faster because models were loaded. I measured cold, warm and cached in one process: warm was actually slower because of a Gemini 503 and a non-deterministic plan, while the answer cache made repeats 300 times faster. It taught me to separate one-time costs from per-request costs and to expect LLM latency variance."

---

## Story 4: Users were being billed for the system's failures

**Problem (found in a code audit).** Usage was recorded in a `finally` block, so a failed request, or a message the input guardrail rejected before any model call, still used up the user's daily allowance (2 messages per day). A failed first message also left an empty chat session behind.

**Why.** A `finally` runs on every path; the intent "count what the user consumed" was implemented as "count every attempt".

**Fix.** Record usage only for a real answer, or for a blocked answer that did spend tokens. Create the session only after the graph succeeds.

```python
if not blocked or usage["prompt_tokens"] + usage["output_tokens"] > 0:
    quotas.record_usage(conn, user_id, usage["prompt_tokens"], usage["output_tokens"])
```

Tests cover four cases: failed run, blocked before any model call, blocked after spending tokens, and no empty session.

**30-second version.** "An audit showed that failed and rejected requests still burned a user's daily quota, because usage was recorded in a `finally`. I moved it to record only successful work, plus blocked requests that had actually cost tokens, and made session creation happen only after success."

---

## Story 5: The evaluation was measuring the cache, not the pipeline

**Problem.** The offline evaluation calls the same graph as production. The graph has a two-level answer cache, so a repeated golden question could return a stored answer, and the scores would measure the cache instead of the current code. Regressions would be hidden.

**Fix.** A `use_cache` flag in the graph state (default on). The evaluation passes `use_cache=False`, which skips both cache reads **and** writes, so eval runs neither read stale answers nor pollute the chat cache.

```python
if state.get("history") or state.get("notes") or not state.get("use_cache", True):
    return {**state, "cache_hit": False}
```

**30-second version.** "Evaluation shares code with production, and the production answer cache would have made eval scores measure the cache. I added an explicit switch so evaluation bypasses cache reads and writes entirely, which keeps eval results comparable across versions."

---

## Story 6: Guardrail false positives on the project's own topic

**Problem.** The corpus includes an AI-security paper, but the prompt-injection guardrail blocked any text containing "system prompt", "jailbreak", "act as a" or "you are now". Legitimate questions ("What is a system prompt leak?", "How does jailbreaking work?", "the layers act as a feature extractor") would have been refused.

**Why.** Keyword matching cannot tell a **mention** of an attack from an **attempt**.

**Fix.** Match commands aimed at the assistant, not topics. Role-play phrases only count at the start of a sentence or clause; "reveal/print your system prompt" is matched as a command. True positives stay covered ("Ignore previous instructions", "Please print your system prompt", "Enable developer mode").

```python
_CLAUSE_START = r"(?:^|[.!?;:\n]\s*|\band\s+|\bthen\s+)(?:please\s+)?"
INJECTION_PATTERNS = [
    _CLAUSE_START + r"act as (?:if you (?:are|were)|an?)\s",
    r"\b(?:reveal|show|print|repeat|display|output|leak|tell me)\s+(?:me\s+)?(?:your\s+(?:system prompt|instructions|prompt)|the system prompt)\b",
    ...
]
```

Tests assert both directions: benign questions about attacks pass, and real attacks are still blocked.

**Honest limits.** Regex guardrails are a first line, not a complete defence. Because role-play phrases only match at the start of a sentence or clause, a phrasing such as "Also, please act as..." would slip past (a known gap, not covered by a test). The live guardrail check is still to be run.

**30-second version.** "My injection filter blocked questions about the very AI-security paper in my corpus, since keyword rules can't tell mentioning an attack from attempting one. I anchored patterns to command position and added tests in both directions, so it blocks 'print your system prompt' but not 'what is a system prompt leak?'."

---

## Story 7: Graceful degradation instead of stack traces

**Problem.** If every model in a tier failed, or Neo4j was unreachable, users saw a raw exception (Streamlit trace) or a bare 500 from the API.

**Fix.** One domain error, `ServiceUnavailable`, raised by the chat handler for exactly two known causes: all models in the tier failed, and Neo4j unreachable. The UI shows a friendly warning; the API returns **503 with `Retry-After: 60`**. The user is not charged. Unrelated exceptions still propagate, so real bugs are not disguised as outages (there is a test for that).

**30-second version.** "I mapped known infrastructure failures to a single 503 with Retry-After and a friendly message, kept the user's quota untouched, and made sure genuine bugs still surface instead of being hidden as outages."

---

## Story 8: Capacity limits for a free-tier model budget

**Problem.** Open sign-up plus a free Gemini tier means a few users could exhaust the day's quota, and simultaneous heavy requests could pile up.

**Design (confirmed with the owner).**

| Limit | Default | Behaviour |
|---|---|---|
| Messages per user per day | 2 (admins 500) | quota error |
| Different users per day | **20** | a **new** user is refused when full; someone already served today is never cut off; admins exempt |
| Requests processed at once | **3** | non-blocking: at capacity the user gets "busy, retry in a few seconds" (429 + `Retry-After`), and is not charged |
| Global requests per day | 50 | backstop |
| Sign-ups per hour | 10 | throttle |

The concurrency gate is a small in-memory, non-blocking limiter with a context manager that always releases its slot, including on errors. It is tested with real threads (8 workers, 3 slots: exactly 3 enter and 5 are refused).

**Honest limits.** It is per process (the Streamlit app and the API each have their own 3 slots); cross-process limiting would need Redis, which was deferred on purpose. The 50/day global cap is optimistic for the free tier.

**30-second version.** "I designed layered limits: per-user, per-day distinct users, concurrency, global cap and sign-up throttle. The concurrency gate is non-blocking, so overload fails fast with a Retry-After instead of queueing threads, and it's tested with real threads."

---

## Story 9: Test strategy: isolation, real-database tests, and what they caught

**Problems.**
- Unit tests could quietly hit real services and spend real API quota.
- Fake database connections accept SQL a real database rejects (upserts, `<=>` vector search, `percentile_cont`, JSONB casts, cascades).
- One flat `tests/` folder is hard to map onto CI stages.

**Fixes.**
1. **A guard that blocks real services in unit tests.** An autouse fixture patches `psycopg.connect`, the Neo4j driver and `genai.Client` to raise a clear error. A test that forgets to fake one fails loudly instead of silently spending quota.
2. **Integration tests on real Postgres** in a throwaway schema `it_<random>` that is dropped afterwards (only names starting `it_` can ever be dropped). 33 tests. Only the model and the embedding call are faked.
3. **Faker with a fixed seed** for realistic data values (names, emails, passwords) and property-style tests (random emails must always be redacted; quota decisions checked against an independent re-implementation).
4. **Folders by what a test *needs*, not when it runs:** `tests/unit/` (always), `tests/integration/` (`--run-integration`), `tests/live/` (`--run-live`, spends quota). The CI pipeline decides when each runs. The folder marker is applied automatically.

**What the tests actually found.**
- The MLflow 3.16 local file store now raises unless `MLFLOW_ALLOW_FILE_STORE=true`. A real eval run would have failed the same way. Found by the very first full unit run.
- The first run of the integration suite passed 22/22, the first time the application SQL touched a real database.
- While writing an integration test I noticed the health report counted answers with no `cache_hit` field as "not cached", skewing the rate. Fixed in the SQL and pinned by a test.

**Numbers.** 529 unit + 33 integration + 3 live = 565 tests; about 1.5 to 3 minutes for the unit suite (mostly importing torch).

**Suggested CI mapping.** Push/PR runs unit (plus integration with a pgvector Postgres service); nightly runs everything non-live plus a dependency-drift check; weekly/manual runs live and an eval on the golden set.

**30-second version.** "I split tests by what they need: unit tests are fully faked and guarded so they can't reach real services, integration tests run the real SQL in a throwaway schema, and live tests are opt-in because they spend quota. That structure found a real MLflow incompatibility and mapped cleanly onto CI stages."

---

## Story 10: Online evaluation: watching quality in production

**Question it answers.** Offline evaluation (golden set before release) tells you it is safe to ship. Online evaluation tells you it **still works**, on questions you never wrote.

**Design.**
- **Feedback:** thumbs up/down under each answer (UI and API). One rating per user per answer (upsert), owner-only, optional comment PII-redacted and capped at 500 characters.
- **Sampled LLM judge:** about 10% of live answers (`ONLINE_EVAL_SAMPLE_RATE`, 0 = off) are scored 0 to 1 for **faithfulness** (supported by the retrieved context) and **relevance**, by an evaluation-tier model in a **background thread after the reply is stored**. So it never delays the user, is never charged to them, and cannot fail a turn. Blocked and empty answers are skipped.
- **Judge-prompt injection defence:** question, context and answer are fenced as untrusted data, and the judge is told never to follow instructions inside them. Malformed or out-of-range replies are dropped, not stored.
- **Health data:** each stored answer records latency, tokens, cache-hit and blocked flags.
- **Report:** summary (avg/p95 latency, cache-hit and block rates, satisfaction, judged count, mean faithfulness and relevance), **alerts with minimum sample sizes** (so 3 answers cannot cause a false alarm), and a **review queue** of low-scored or thumbs-down answers with the redacted question. That queue is the production-to-golden-set feedback loop.

**Honest limits.** The judge has never run against a real model (its tests fake the model). An LLM judge can be biased or wrong, so scores are a trend, not truth (the judge model is stored with each score). There is no topic-drift monitoring, and nothing is added to the golden set automatically; a human writes the reference answer.

**30-second version.** "For production I sample about 10% of answers into a background LLM-judge for faithfulness and relevance, collect thumbs feedback, and expose alerts and a review queue of the worst answers, which feeds new cases into the offline golden set. The judge runs after the reply, so it never adds latency or cost for the user."

---

## Story 11: Model tiers with per-model fallback

**Design.** Three tiers, at least two models each, all Google models:

| Tier | Used for | Models (primary, then fallback) |
|---|---|---|
| fast | planner, quality check, summaries, simple answers | gemini-3.6-flash, gemini-3.5-flash-lite |
| reasoning | complex answers | gemini-3.8-flash, gemini-3.7-flash |
| evaluation | LLM judges (offline and online) | gemini-3.7-flash, gemini-3.5-flash |

- Each task maps to a tier (`TASK_TIERS`). Nothing hard-codes a model name.
- **A tier never borrows another tier's models.** A double failure raises an error, so cost and quality never change silently.
- **Cooldowns are per model** (quota 15 min, auth 1 h, unavailable 60 s), because free-tier quotas are per model.
- The planner labels each sub-question `simple` or `complex` at no extra model call, so the answer step picks the fast or the reasoning tier. A missing or invalid label counts as `complex`.
- Provider details live in one module. Only key **names** are configured, never key values; switching a tier to another vendor is a settings edit.

**Evidence it works.** Failover to the tier's second model happened in both live runs.

---

## Story 12: Developer environment: 1 GB of RAM in the editor

**Problem.** On a 7 GB machine the editor was using about 2 GB. Measuring per process showed the Pylance language server alone at **1,064 MB**, because nothing told it to skip the virtual environment (torch, transformers, docling) and it was analysing the parent folder holding several projects.

**Fix.** Workspace settings to exclude `venv*` and data folders, check open files only, and stop indexing. Pylance dropped to **319 MB**. Total editor memory barely moved because other processes grew back after the reload (extension host, editor window, extensions), which is a good reminder to measure the whole and not one part.

---

## Story 13: Probing the guardrails live found four gaps the unit tests could not

**Situation.** Unit tests said every guardrail worked. I then ran four real probes through the live pipeline (two attacks, two legitimate-but-suspicious questions) and saved a report. The attacks were blocked in under 0.1 s with zero model calls. The other two questions exposed real gaps, and a follow-up test exposed a third and fourth.

| # | What the live run showed | Why | Fix |
|---|---|---|---|
| 1 | A correct answer about "system prompt leaks" (from the AI-security paper) was flagged `possible_injection_leak` | The **output** guardrail had the same bug as the input one: bare words "system prompt" instead of the model talking about *its own* instructions | Match first-person leaks ("my system prompt", "here are my instructions"); tests in both directions |
| 2 | "I was just diagnosed with lung cancer. Should I start chemotherapy?" got no warning and no safety note | The input guardrail **detected** the clinical-advice framing, but nothing consumed the result: it was computed and thrown away | Turn it into a `medical_advice_framing` flag and put a fixed, non-LLM safety note at the top of the answer |
| 3 | The model wrote citations as `[file.pdf, p. 15]` and `[file.pdf, p. 1, 8]`; the checker extracted **nothing** from them | The regex only knew `p.15`; any other format silently escaped verification, so a fabricated citation in that format would never be flagged | Regex accepts the formats actually seen (space after `p.`, bare numbers after commas, `pp. 3-5`, en dash); parametrized tests including the exact real strings |
| 4 | A saved note was recalled on the first question but not on the follow-up | Long-term recall embedded only the literal follow-up ("And which of them was the most accurate?"), which is far from the note (cosine distance 0.576 against a 0.5 limit) | Look up notes with the user's previous question plus the follow-up (distance 0.207) |

**Details worth mentioning.**
- **Safety note and the cache.** The medical note must appear even on a cache hit: a cached answer may have been stored for a neutral phrasing of the same question, so a medical user would otherwise get it with no note. It is added *after* the answer is cached, so the shared cache never stores it, and a check prevents adding it twice.
- **I measured before fixing recall.** With the real embedding model (local, no API cost) I compared distances: on-topic questions 0.18 to 0.27, off-topic 0.42 to 0.58. That confirmed the fix, and showed the 0.5 threshold was already loose: two off-topic questions asked on their own (0.42, 0.49) were recalled.
- **Honest replay.** I re-ran the saved real answers through the fixed guardrails at no API cost. The false leak flag disappeared and the medical flag appeared. The saved report did not store retrieved passages, so I could not replay citation verification against real context and did not claim to.

**30-second version.** "Unit tests passed, so I probed the guardrails live and found four gaps: a false positive on the output side, a clinical-advice detector whose result was computed and then discarded, a citation checker that silently ignored two citation formats the model actually uses, and a memory lookup that failed on follow-ups. The lesson: a guardrail that detects but isn't wired to an action, or a verifier that skips inputs it doesn't recognise, fails silently, so I test them against real model output."

**Still open from this story.** The recall distance threshold (0.5) looks too loose: on a small sample of 6 questions, 0.35 would separate on-topic from off-topic. I did not change it on so little data. The citation check confirms only that a cited page was *retrieved*, not that the claim is actually on it.

---

## What I got wrong and corrected (worth telling)

Rigor is a strength; these are real corrections made during the work:

1. **Said the caching failure happened on every call.** Wrong: it happens once per model + instruction per process (about 3 wasted requests per start). Corrected after re-reading the code.
2. **Called the two sub-questions' context "the same 9 chunks".** That was inferred from equal token counts, not verified. The merge logic therefore measures overlap directly instead of assuming it.
3. **Reported "513 unit + 33 integration tests".** 513 was the total passed at that moment. Verified count by collection: 502 unit + 33 integration + 3 live.
4. **Said "one paper has no entity links" in the graph.** Verifying in the database showed **two** (a lung-microbiome paper and the Bhutan Sentinel-2 paper).
5. **Assumed a second run would be faster.** Measured: it was not (Story 3).

---

## What is not fixed or not proven (say this if asked)

- The single sub-question was still labelled `complex`, so a single-paper lookup used the reasoning tier. The prompt wording did not move that label on this question; not tuned further on one example.
- Time-to-503: the planner call took about 30 s including a failed request and failover, and we do not log how long the failed request took to return the 503.
- Retrieval took about 20 s on the first question of a process (cold model load); a long-running server pays it once.
- Two papers have no method/dataset/metric links in the graph (extraction produced nothing usable). This affects graph-style questions ("which papers use X"), not text retrieval.
- Online judge and the overlap merge have not been exercised against a live model.
- The full offline evaluation (36 questions) has not been run: it would exceed the free daily quota.
- Langfuse tracing is off (its Docker stack was wiped); the code degrades gracefully with `LANGFUSE_DISABLED=1`.

---

## Numbers cheat sheet

| Fact | Value |
|---|---|
| Tokens per question | 32,015 to 14,609 (-54%) |
| Model calls per question | 5 to 3 |
| Pipeline latency | 86.9 s to 58.2 s (-33%) |
| Answer cache hit | 0.17 s, 0 model calls |
| Cold vs warm | 57.7 s vs 65.8 s (warm slower: 503 + plan variance); retrieval 17.0 s to 5.2 s |
| Failover cost | up to about 14 s of waiting removed per busy call (one 2 s wait instead of 2+4+8 s) |
| Tests | 565 (529 unit, 33 integration, 3 live); 3 live smoke tests passed (real Gemini, Neo4j, full pipeline) |
| Golden set | 36 hand-written questions, 3 per paper, DVC-versioned |
| Graph | 12 papers, 316 sections, 174 images |
| Capacity | 3 concurrent, 20 users/day, 2 messages/user/day, 50/day global |
| Editor memory | Pylance 1,064 MB to 319 MB |

---

## Likely questions and short answers

- **"How did you find the performance problem?"** Instrumented per stage and per model call, then read the table: the same context was sent four times.
- **"How do you know it improved?"** Same question, same instrumentation, before and after: 5 to 3 calls, -54% tokens, -33% latency. It is one run, so I treat it as directional, and I have unit tests for the mechanisms.
- **"How do you handle LLM provider failures?"** Tiered models with in-tier fallback, error classification by status (429 vs 5xx vs auth), per-model cooldowns, a fast failover when a healthy fallback exists, and a clean 503 when the whole tier is down.
- **"How do you evaluate a RAG system?"** Offline golden set with retrieval and answer metrics as a release gate, plus online sampled judging, user feedback and health alerts, joined by a review queue that feeds new cases back into the golden set.
- **"What would you do next?"** Run the full offline evaluation once quota allows, set a baseline, add an eval gate to CI, and fix the complexity-label tuning using evaluation data instead of single examples.
- **"What are the risks in your design?"** LLM-as-judge bias, regex guardrails as a first line only, in-memory concurrency (per process), and free-tier quota limits.
