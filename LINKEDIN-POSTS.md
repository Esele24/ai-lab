# LinkedIn drafts — one per project

Written 2026-08-11 from what was actually built and measured today. Every number in
these posts is real and appears in `README.md` or `model/spam_report.json`.

**Before posting, three things:**

1. **Nothing is deployed.** There is no link to put in these. Either post them as-is
   (they stand on the detail), or deploy first and add a URL.
2. **Read each one in your own voice.** These follow your documented style — concise,
   direct, no fluff — but I have not put opinions in your mouth. If a line does not
   sound like you, cut it; do not soften it.
3. **Don't post all seven in a week.** One every 2–3 days. Post 3 (the trained model)
   is the strongest — lead with it.

Suggested order: **3 → 2 → 6 → 1 → 5 → 7 → 4**. Strongest proof first, the honest one
about limits (7) once you have credibility banked, and the architecture post (4) last
because it only lands after people have seen the pieces.

---

## Post 3 — Custom Trained Model ⭐ post this first

> My first model scored 96.95% accuracy.
>
> It was missing a quarter of all the spam.
>
> I trained a spam classifier today — TF-IDF and logistic regression written out in
> numpy, no scikit-learn. 5,574 real hand-labelled SMS messages. Test set held out
> before a single weight was updated.
>
> 96.95% accuracy looked fine until I checked recall: **76.64%**. Nearly a quarter of
> real spam sailed through.
>
> Two fixes:
>
> 1. Only 13.4% of the dataset is spam. So the fastest way to lower the loss is to lean
> towards "not spam" — which reads as high accuracy and terrible recall. I weighted each
> spam example by the class ratio, so a missed spam costs as much as a false alarm.
>
> 2. Every phone number in the data is a token seen once, so it gets pruned and the
> model learns nothing from it. I collapsed every number into one symbol and every link
> into another. Those two hand-made features came out as the **top two spam indicators**
> in the trained model.
>
> Final: **99.01% accuracy, 97.08% recall, 96.03 F1.**
>
> The number I keep on the page next to all of them: **87.71%**. That is what you score
> by guessing "not spam" every single time.
>
> An accuracy figure without its baseline next to it is not telling you anything.

---

## Post 2 — AI Data Analysis

> Most "chat with your CSV" demos have the model write pandas code, then run it.
>
> That is a remote code execution hole and a liar in the same feature.
>
> A liar because when its generated code errors, the model will still state a number.
> It has no way to know the difference between a total it computed and a total it
> produced.
>
> I built mine the other way round today.
>
> The model never sees your rows. It gets the column names, the types and three sample
> values, and its only job is to return a **structured operation** — which column, which
> aggregation, which filter, which chart — locked to a strict schema at temperature 0.
>
> Pandas does the arithmetic. All of it.
>
> Then a second call writes the sentences, and it is handed *only the computed result*.
>
> So the model cannot report a total that pandas did not compute. The worst it can do is
> pick the wrong column — and the chosen operation is printed beside every answer, so
> you can see it happen.
>
> Tested it on a sales file. It grouped revenue by category correctly, built a monthly
> trend, and when I asked for customer email addresses it said: *"The dataset does not
> contain customer email addresses."*
>
> Refusing is a feature. It is most of the work.

---

## Post 6 — Industry-Specific AI Helper

> A general AI contract reviewer can only react to text that is in front of it.
>
> Which means it will never tell you what your contract is missing. And what is missing
> is usually the expensive part.
>
> I built a document reviewer today where each industry carries its own clause
> checklist — legal, oil and gas, real estate, healthcare, e-commerce, education. Every
> item gets assessed, including the ones the document never mentions.
>
> I tested it on a contract I wrote to be quietly awful: 90-day payment terms with no
> late interest, the contractor can walk on 7 days' notice while the client is locked in
> for 24 months, and the client indemnifies the contractor for the contractor's own
> negligence.
>
> 10 findings against 10 checklist items. 5 absent. 3 weak.
>
> The part I care about most is what happens *after* the model answers.
>
> Every finding has to quote the document verbatim. Then code checks that quote is
> actually in the source text. Anything it cannot show gets dropped — and a finding that
> claims a clause is present but produces no quote is demoted to **absent**.
>
> A review that cites a clause your contract does not contain is worse than no review.
> It is the one you would have acted on.

---

## Post 1 — Smart Document Assistant

> The hard part of "chat with your PDF" is not the chat.
>
> It is making the thing admit when the answer isn't there.
>
> Built a document assistant today. Upload a PDF, ask a question, get an answer that
> names the page it came from.
>
> Three decisions that did the work:
>
> **Chunks split on sentence boundaries, not character counts.** Cutting mid-sentence is
> the classic failure — the retrieved excerpt reads as a fragment, and the model closes
> the gap by inventing. There is 150 characters of overlap so a fact sitting on a chunk
> boundary survives whole somewhere.
>
> **The page number travels with the chunk** from extraction all the way to the answer.
> A citation you cannot follow is worth very little when you are checking a contract.
>
> **The model only ever sees the retrieved excerpts**, with a rule saying it may not go
> beyond them.
>
> So when I asked it the capital of Brazil, it said: *"That isn't in the documents you
> uploaded."*
>
> It knows the capital of Brazil. That is exactly the point.

---

## Post 5 — Smarter Prompt System

> If you use an AI to judge which of two prompts is better, it will lie to you in two
> specific ways.
>
> It prefers the longer answer. And it prefers whichever one it was shown first.
>
> Both are measurable, and most A/B prompt tools inherit both and call it a result.
>
> Built one today that handles it:
>
> — Which real output gets labelled "A" is **shuffled** before the judge sees them, so
> position bias stops being systematic
> — It scores against four named criteria instead of "which is better"
> — Each output's **word count is printed beside its score**, so a win that is really
> just verbosity is visible on the page
>
> Then the test output caught a bug I had put in myself.
>
> The judge wrote *"Output A generated the message, whereas Output B…"* inside its
> explanation. But that sentence is in the **judge's** label space — and I flip the
> labels back afterwards. So on every shuffled run, the explanation contradicted the
> verdict.
>
> Fixed by banning the letters A and B from the prose entirely and making it describe
> the winner's property instead.
>
> The shuffle was the right idea. I just forgot that the model writes in the coordinates
> I gave it, not the ones I show the user.

---

## Post 7 — AI Voice Agent

> I built an AI voice agent today. It is not a phone agent, and I am not going to call
> it one.
>
> Every voice agent post you have seen this month shows a dashboard with inbound calls,
> call routing and live transfers. That needs Twilio or Vapi, a bought phone number and
> per-minute billing. I have none of those, so I did not build that.
>
> What I did build, and what runs: you speak into the browser, it answers out loud, and
> when it has your name, number, date and service it writes a real booking to the
> database.
>
> The rules I gave it are mostly about what it may not do:
>
> — **Never invent a price.** It has no price list. Asked what catering costs, it
> collects your details and says a coordinator will quote.
> — **Never say a date is available.** It cannot see a calendar.
> — **Never claim it sent an email.**
> — Replies capped near 40 words, no markdown — a speech synthesiser reads asterisks out
> loud, and nobody on a call follows a paragraph.
>
> And the booking is written by code *after* it checks the four required fields. Not by
> the model announcing that a booking happened.
>
> That is the difference between an agent that books and an agent that says it booked.
>
> The transport is the easy half. If I attach a phone number tomorrow, none of the agent
> logic changes.

---

## Post 4 — Full AI-Powered App (architecture)

> Halfway through building today, pip gave up.
>
> `scipy` is a 36 MB wheel. My connection was doing 74 kB/s. Five timeouts, then a file
> lock, then nothing. Which killed the plan — Streamlit and scikit-learn were the whole
> stack.
>
> So I rebuilt it with what was already on the machine.
>
> The backend is Python's own `http.server` and `sqlite3`. The frontend is HTML, CSS and
> vanilla JS — no framework, no build step, no CDN. Instead of scikit-learn I wrote the
> TF-IDF and the logistic regression out by hand in numpy. Instead of FAISS, the vector
> search is about 60 lines.
>
> That last one I would now choose on purpose. FAISS earns its keep past a million
> vectors. A 300-page PDF is a few thousand — an exact dot product is sub-millisecond.
> FAISS there is a dependency that buys nothing and hands back an *approximate* answer
> where an exact one was free.
>
> Seven AI tools, one server file, one database, one design system. Total new
> dependencies installed: **zero**.
>
> A bad connection made me learn what each of those libraries was actually doing for me.
> Turns out for this size of problem, three of them were doing very little.

---

## Optional post — the honest one about the source

Only post this if you want to. It is the most engagement-shaped of the eight, and it
takes a swipe at a genre of content, which cuts both ways.

> A carousel went round called "7 AI projects that will get you hired in 2026."
>
> I built all seven today. Here is what the slides left out.
>
> **Three of the seven were the same project.** "Document assistant", "industry-specific
> AI helper" and "full AI-powered app" are one RAG pipeline with different placeholder
> text. Building them separately teaches you one thing three times.
>
> **The trained model is the only one that is actually hard**, and it is the one every
> post skips past. It is also the only one that still worked when the API quota ran out
> — because it runs locally and needs no API at all.
>
> **The voice agent cannot be built as shown** without a phone number and per-minute
> billing. Mine runs in the browser. I say so on the page.
>
> **Every slide says "OpenAI API".** I don't have an OpenAI key. The whole thing runs on
> the free tier of a different provider, which is capped at about twenty requests — my
> own test suite exhausted it.
>
> The projects are fine. The slides are just the easy 20%.
>
> The 80% is what happens when the connection dies, the quota runs out, and the model
> confidently tells you something that isn't in the document.
