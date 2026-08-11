# Shipping AI Lab — what is done, and the three logins that are not

**Status 2026-08-11:** the code is deployment-ready and verified locally. Deployment
itself is blocked on credentials only you can supply. Nothing below is guesswork — the
code changes it needs are already made and tested.

---

## Why it is two deployments, not one

Vercel cannot host the Python backend. Not a limitation of the free plan — the vector
index and the loaded dataframe live in the process, and Vercel functions are stateless
and short-lived. So:

```
  Browser  ──►  Next.js on Vercel  ──►  Python on Render
                (rewrite proxies /api/*)   (numpy, pandas, sqlite)
```

The browser only ever talks to Vercel. `next.config.ts` rewrites `/api/*` to
`BACKEND_URL` **server-side**, which is why there is no CORS to configure and the
backend URL never ships to the client.

---

## The three blockers

| # | Blocked on | Why I could not do it |
|---|---|---|
| 1 | **Vercel login** | The CLI is installed (v55.0.0) but not authenticated — no auth file, and `vercel whoami` fails. `vercel login` needs an interactive browser/email confirmation. |
| 2 | **Render account** | No Render CLI and no `RENDER_API_KEY`. Free-tier signup is a browser flow. |
| 3 | **A GitHub repo** | `gh` IS authenticated as `Esele24`, and Render deploys from Git — but this workspace is not a git repo, and pushing your code to a remote publishes it under your name. That is your call, not mine, especially in a folder holding a live API key. |

---

## Step 1 — the repo (5 min)

⚠️ **Check `.gitignore` before the first commit.** `ai-lab/.env` holds your real Gemini
key. It is already ignored, but verify with `git status` that `.env` does **not** appear.
A key in a commit stays in the history even after you delete the file.

```powershell
cd "C:\Users\ESELE OKOGBO\Desktop\CLAUDE\ai-lab"
git init
git add -A
git status          # <-- CONFIRM .env is NOT listed before going further
git commit -m "AI Lab backend: seven AI projects, one Python server"
gh repo create ai-lab-backend --private --source=. --push
```

Keep it **private**. Render deploys from private repos on the free plan, and nothing
here needs to be public.

## Step 2 — the backend on Render (10 min)

1. Sign up at render.com (free, GitHub login).
2. **New → Blueprint**, pick `ai-lab-backend`. It reads [render.yaml](render.yaml).
3. When prompted for the secret, paste `GEMINI_API_KEY` — the value from
   `tender-tool/.env.local`. Everything else is already set in the blueprint.
4. First build runs `pip install -r requirements.txt && python train_model.py`, so
   project 03 is trained on the server. Expect 3–6 minutes.
5. Copy the service URL, e.g. `https://ai-lab-backend.onrender.com`.

**Confirm it works before touching Vercel:**

```powershell
curl https://ai-lab-backend.onrender.com/api/dashboard
```

You want `"model_ready": true` and `"key_present": true` in the response.

## Step 3 — the frontend on Vercel (5 min)

```powershell
cd "C:\Users\ESELE OKOGBO\Desktop\CLAUDE\ai-lab-web"
npx vercel login          # interactive — this is blocker #1
npx vercel link
npx vercel env add BACKEND_URL production
#   paste: https://ai-lab-backend.onrender.com   (no trailing slash)
npx vercel deploy --prod
```

Same pattern as your other five sites: Vercel CLI, not GitHub-connected, so redeploy
with `vercel deploy --prod` from this folder.

## Step 4 — check it end to end

Open the Vercel URL and confirm, in this order, because they fail differently:

1. **Dashboard** shows non-zero stats → the rewrite reached Render.
2. **03 Model** shows 99.01% and classifies a message → the trained model survived the
   build step. This one needs no API quota, so it works even when the model is rate
   limited.
3. **01 Documents** → upload a small PDF and ask a question. This is the one that spends
   quota.
4. Toggle dark mode, then reload. The theme must not flash.

---

## ⚠️ Four things that will look like bugs and are not

**1. The first request after a quiet spell hangs for 30–60 seconds.** Render's free
instance sleeps after ~15 minutes idle. If you put this link in a LinkedIn post, the
first visitor each hour waits. Consider saying so on the page, or upgrade the plan.

**2. The activity log, saved prompts and bookings reset on every deploy.** Render's free
filesystem is ephemeral, so `data/ai_lab.db` and `data/doc_index/` are wiped. Fine for a
demo; `core/store.py` is the seam to swap for Supabase when it stops being fine.

**3. Everything AI returns 429 after about 20 requests.** That is the Gemini free tier,
not your code. `check_quota.py` tells you which of the two 429s you have.

**4. The rate limit is deliberate.** 30 AI requests per hour per IP, set by
`RATE_LIMIT_PER_HOUR`. It exists because a public URL plus your API key plus a ~20-request
free quota means one stranger with a loop can spend your whole day's allowance.
✅ Verified with `test_ratelimit.py`: the model routes cap correctly, while the local
classifier and all GET routes stay uncapped.

---

## What was changed to make this deployable

Worth knowing, because these would each have been a failed deploy:

- **`server.py` now reads `PORT` and `HOST` from the environment.** It bound
  `127.0.0.1:8000` hard-coded, which on Render means the health check never connects and
  the deploy fails with no useful error. Local default is still `127.0.0.1`, so running
  it at home does not expose it to your network.
- **Per-IP rate limiting** on the seven routes that cost money. Uses a deque of
  timestamps per IP rather than a counter — a counter cannot expire, so it either never
  resets or resets everyone at once.
- **`X-Forwarded-For` handling**, taking the *first* entry. Behind Vercel→Render every
  request arrives from the proxy's IP, so without this the whole internet shares one
  rate-limit bucket.
- **`requirements.txt`** pinned to the five versions verified locally.
- **`render.yaml`** with training in the *build* step. Training at request time would
  exceed the request timeout, and project 03's page would render "not trained".
