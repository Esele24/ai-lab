# AI Lab — seven AI projects, one application

Built 2026-08-11 by Esele Okogbo, Port Harcourt.

Seven projects from a widely-shared *"7 AI projects that will get you hired in 2026"*
carousel, built honestly — including the parts that could not be built as advertised,
which are labelled as such on their own pages rather than hidden.

---

## Run it

```powershell
cd "C:\Users\ESELE OKOGBO\Desktop\CLAUDE\ai-lab"

python check_setup.py     # proves the key, the chat model, schema mode and embeddings
python train_model.py     # trains project 03. ~1 minute. Only needed once.
python server.py          # then open http://localhost:8000
```

### Two front ends

There are two, both working, and they talk to this same backend:

| | Where | How to run |
|---|---|---|
| **Next.js 16 + Tailwind v4** — matches the rest of the portfolio | [`../ai-lab-web`](../ai-lab-web) | `npm run dev` (with `server.py` running), then port 3000 |
| **Vanilla HTML/CSS/JS** — zero dependencies, served by `server.py` itself | `web/` | nothing extra; it is on port 8000 |

The Next.js one is the deployable, better-looking one. The vanilla one still works and is
the fallback that needs no Node at all. Deployment: [DEPLOY.md](DEPLOY.md).

Verify everything at once (**makes ~20 model calls — this alone exhausts the free tier**):

```powershell
python test_all.py        # in a second terminal, with the server running
python check_quota.py     # one cheap call, to see if the rate limit has cleared
```

**Dependencies: none to install.** `numpy`, `pandas`, `pypdf`, `python-docx` and
`requests` were already on the machine. Everything else is Python standard library —
`http.server`, `sqlite3`, `json`, `base64`, `re`. The frontend has no framework, no
build step and no CDN.

---

## Why the stack looks like this

The original plan was Streamlit + scikit-learn, exactly as the carousel specifies.
It failed on the connection: `scipy` is a 36 MB wheel and pip timed out five times
at ~74 kB/s before giving up.

That constraint turned out to be a straight upgrade:

| Carousel says | What is here | Why it is better, not just different |
|---|---|---|
| OpenAI API | Gemini via raw `requests` | **There is no OpenAI or Anthropic key on this machine.** The Gemini key exists and works. Raw HTTP means you can read the actual call. |
| FAISS | ~60 lines of numpy | FAISS earns its keep past a million vectors. A 300-page PDF is a few thousand — an exact dot product is sub-millisecond, so FAISS would be a dependency that buys nothing and returns an *approximate* answer where an exact one was free. |
| scikit-learn | TF-IDF + logistic regression by hand | `model.fit()` teaches nothing. Each is about thirty lines, and writing them means every number on the page is explainable. |
| Streamlit | `http.server` + vanilla JS | One backend file you can read end to end, and it deploys anywhere Python runs. |
| Plotly | CSS bar and line charts | No runtime dependency to ship over Nigerian mobile data. |

---

## The seven

| # | Project | Status | The one thing worth knowing |
|---|---|---|---|
| 01 | Smart Document Assistant | ✅ working | Answers cite the page. Refuses questions the documents cannot answer, in those words. |
| 02 | AI Data Analysis | ✅ working | The model picks the *operation*; **pandas computes every number**. No `exec()`, anywhere. |
| 03 | Custom Trained Model | ✅ working | 99.01% accuracy, 96.03 F1 on a held-out test set. Trained here, in numpy. |
| 04 | Full AI-Powered App | ✅ working | The dashboard *is* this project — shared client, shared DB, real activity log. |
| 05 | Smarter Prompt System | ✅ working | The A/B judge sees the two outputs in **shuffled order**, because judges have position bias. |
| 06 | Industry-Specific Helper | ✅ working | Per-industry clause checklist, so it can report what is **missing**. Every quote is verified against the source. |
| 07 | AI Voice Agent | ⚠️ **browser, not phone** | Real speech in and out and a real booking written to SQLite. **No telephony** — that needs Twilio/Vapi and paid minutes. Said plainly on the page. |

---

## 01 · Smart Document Assistant — RAG

`core/docs.py` · `core/vectors.py` · `web/doc.html`

1. **Extract** text, *keeping the page number*. A citation you cannot follow is worth
   very little when you are checking a contract.
2. **Chunk** on sentence boundaries, ~1,200 chars with 150 of overlap. Splitting
   mid-sentence is the classic RAG bug — the excerpt reads as a fragment and the model
   fills the gap by inventing. The overlap means a fact on a chunk boundary survives
   whole somewhere.
3. **Embed** each chunk to 3,072 numbers. The question is embedded with
   `RETRIEVAL_QUERY`, documents with `RETRIEVAL_DOCUMENT` — using the same task type
   for both measurably degrades retrieval.
4. **Retrieve** by cosine similarity, and only those excerpts reach the prompt.

**Verified:** indexed a real PDF, asked for the late-delivery penalty, got
`"The penalty for late delivery is 2.5 percent of the contract value per week [1]."`
cited to `contract.pdf p.1`. Asked the capital of Brazil, got
`"That isn't in the documents you uploaded."` A `.pdf` that was not a PDF was rejected
by magic-number check.

**Known limits, on the page too:** no OCR (a scanned PDF has no text layer); DOCX has
no page numbers until rendered, so it cites the filename only — a guessed page number
is worse than none.

## 02 · AI Data Analysis — the decision that matters

`core/analysis.py`

The obvious build is "ask the model to write pandas, then `exec()` it". That is two bad
things at once: a remote-code-execution hole, and a liar — when its own code errors the
model states a number anyway.

Here the model returns a **structured operation** under a strict `responseSchema` at
temperature 0: which column, which aggregation, which filters, which chart. **Pandas
does all the arithmetic.** A second call writes the prose and is given *only the
computed result*, never the rows.

So the model **cannot** report a total pandas did not compute. Its worst available
failure is picking the wrong column — visible, because the chosen operation is shown
beside every answer.

**Verified:** grouped revenue by category (Venue Hire, ₦6,700,000), built a 6-point
monthly time series, and correctly refused *"what is each customer's email address?"*
with *"The dataset does not contain customer email addresses."*

## 03 · Custom Trained Model — the real one

`core/mlmodel.py` · `train_model.py`

Dataset: **UCI SMS Spam Collection**, 5,574 real hand-labelled messages, 13.4% spam.
4,459 train / 1,115 held out. The vectoriser is fitted on the **training half only** —
fitting the vocabulary before splitting leaks test data into training and inflates
every score after it.

| | First run | After two fixes |
|---|---|---|
| Accuracy | 96.95% | **99.01%** |
| Precision | 98.13% | 95.00% |
| **Recall** | **76.64%** | **97.08%** |
| F1 | 86.07% | **96.03%** |

Baseline (always predict "ham"): **87.71%**. Any accuracy quoted without that number
beside it is not telling you anything.

**The two fixes:**

1. **Feature engineering.** Every phone number is a token seen once, so `min_df=2`
   prunes it and the model learns nothing from it. Collapsing all of them into
   `__num__` and all links into `__url__` gives it a feature it can learn from. Those
   two hand-made tokens came out as the **top two spam indicators**. A library cannot
   do this for you — it depends on the data.
2. **Class weighting.** At 13.4% spam, an unweighted model minimises loss fastest by
   leaning towards "ham". That reads as high accuracy and terrible recall — it was
   missing a quarter of real spam. Weighting each spam example by `n_ham / n_spam`
   makes a missed spam cost as much as a false alarm.

Predictions are **auditable**: the page shows each token's TF-IDF value times its
learned weight, so you can see exactly what drove the decision, plus the tokens that
were outside the vocabulary and therefore ignored.

## 04 · Full AI-Powered App

`server.py` · `core/store.py` · `web/index.html`

The dashboard is not a landing page for the others — it is the application they live
inside. Twenty JSON routes, one Gemini client, one SQLite database, one design system.

Every tool writes to `runs` and the dashboard reads it back, so the activity log
survives a restart — which a variable in memory would not. `core/store.py` is the seam
to swap for Supabase later: keep the function names, replace the bodies.

**Security:** static file serving does a path-containment check. Without it,
`/../.env` serves your API key to anyone who asks. There is a test for exactly that.

## 05 · Smarter Prompt System

`core/prompts.py`

Build prompts, save them to SQLite with their placeholders (upsert by name, so editing
replaces instead of duplicating), fill them, and A/B test two.

**The A/B test is the hard part.** A model judging two outputs has two measurable
biases: it prefers the **longer** answer, and it prefers whichever it was shown
**first**. So this one shuffles which real output is labelled "A", scores against four
named criteria instead of "which is better", and prints each output's **word count
beside its score** — so a win that is really just verbosity is visible.

🐛 **A bug the test output caught:** the judge wrote *"Output A directly generated a
message, whereas Output B…"* inside its `reason` — a sentence in the **judge's** label
space, while the UI shows re-mapped labels. On every shuffled run the explanation
contradicted the verdict. Fixed by forbidding the letters A and B in the prose and
making the judge describe the winner's *property* instead.
⚠️ **The fix is applied but was not re-verified under a live call** — the free tier was
exhausted by then. Re-run `python check_ab.py` when quota is back; a run whose `reason`
names a letter means the fix did not hold.

**Verified before that:** a 41-word specific prompt beat a 698-word vague one, with the
judge order shuffled that run.

## 06 · Industry-Specific AI Helper

`core/review.py`

Six industries, each with its own clause checklist — Legal, Oil & Gas, Real Estate,
Healthcare, E-commerce, Education. Every checklist item is assessed, **including the
ones the document never mentions**. That absence is the finding a general chatbot
structurally cannot make, because it only reacts to text that is there.

Three guarantees, the same three as the tender tool:

1. temperature 0
2. strict `responseSchema` — output is a shape, not prose to parse
3. **quote verification** — after the model answers, every finding's quote is checked
   against the actual document text. Anything it cannot show verbatim is dropped, and a
   finding claiming a clause is PRESENT with no quote is **demoted to ABSENT**. A
   finding citing a clause the document does not contain is worse than no finding: it
   is the one you would act on.

**Verified:** reviewed a contract with deliberately bad terms (90-day payment with no
interest, one-sided 7-day termination, client indemnifies contractor for the
contractor's *own* negligence). 10 findings against 10 checklist items — 5 ABSENT,
3 WEAK. All 6 quotes located verbatim in the source; 0 dropped.

## 07 · AI Voice Agent

`core/voice.py` · `web/voice.html`

**A browser voice agent, not a phone agent, and the page says so.** The carousel shows
inbound calls, routing and transfers — that needs Twilio or Vapi, a purchased number
and per-minute billing. None of that exists here, so none of it is claimed.

What does work end to end, free: speech in and out via the browser's Web Speech API
(Chrome/Edge; typing is the fallback elsewhere and the agent still speaks), the model
for conversation and intent, and a **real booking written to SQLite**. The turn logic
in `core/voice.py` is exactly what would carry over if a number were attached —
swapping the transport does not change the agent.

**The booking is written by Python after it checks the four required fields**, not by
the model deciding a booking happened. That is why this agent cannot confirm a booking
that does not exist.

**Forbidden, in the system prompt:** inventing a price (it has no price list),
confirming a date is free (it cannot see a calendar), claiming to have sent an email.
Replies are capped near 40 words with no markdown — a speech synthesiser reads
asterisks out loud.

**Verified:** asked *"how much to cater for fifty people?"* → it refused to quote and
asked for a name instead. A four-turn booking script wrote booking #1 with name, phone,
date, guests and service all correctly extracted.

---

## ⚠️ The free tier is small

```
generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20
```

`test_all.py` makes about twenty model calls and **exhausts the quota by itself**. When
that happens every AI tool returns a 429 while project 03 keeps working perfectly,
because the classifier runs locally and needs no API at all.

`core/gemini.py` parses the API's own `"Please retry in 50.9s"` hint and waits that
long, rather than a fixed 2-second backoff that just burns another refused request.

**Two different 429s, and telling them apart matters:**

- `limit: 0` → this key has **no quota for that model name**. The model is wrong.
  Retrying will never fix it.
- `limit: 20` → a real rate limit. Wait it out.

---

## Test results — 2026-08-11

`24 / 24 checks passed`, covering all seven projects plus the path-traversal check.
The one logged failure in the activity log is intentional: the test that confirms a
`.pdf` which is not a PDF gets rejected.

## Not done

- **Not deployed yet — blocked on three logins, not on code.** The Vercel CLI is not
  authenticated, there is no Render account, and there is no git repo. Everything code-side
  is ready and tested: `PORT`/`HOST` from the environment, `requirements.txt`,
  `render.yaml` with training in the build step, and per-IP rate limiting. Exact steps and
  the four free-tier gotchas: [DEPLOY.md](DEPLOY.md).
- **Single-user.** The loaded dataframe and vector index are in-process, so two people
  hitting it at once would share state. The document index and database are on disk;
  the dataframe is not.
- **No auth.** Locally it binds `127.0.0.1`. Deployed, the only protection is the rate
  limit — **30 AI requests per hour per IP** (`RATE_LIMIT_PER_HOUR`), which exists because
  a public URL plus a live API key plus a ~20-request free quota means one stranger with a
  loop can spend the whole day's allowance. ✅ Verified with `test_ratelimit.py`.
