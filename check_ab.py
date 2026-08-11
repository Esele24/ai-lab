"""Run the A/B comparison three times to confirm the shuffle stays consistent.

The bug this catches: the judge is shown the outputs in random order, and the labels
are re-mapped afterwards. If the judge writes "Output A was better" inside its reason
string, that sentence is in the JUDGE's label space and contradicts the verdict on
every shuffled run. The fix was to forbid the letters in the prose; this proves it.
"""
import requests

A = "Write a message to a business about their website."
B = ("You are writing one WhatsApp message to a Nigerian law firm that has no website. "
     "Under 60 words. Name one concrete thing they are losing. End with a question. "
     "No greeting longer than three words. Do not say I am a developer.")

for run in range(1, 3):
    data = requests.post(
        "http://localhost:8000/api/prompt/compare",
        json={"prompt_a": A, "prompt_b": B,
              "test_input": "The firm is Forcados Partners in Port Harcourt."},
        timeout=300,
    ).json()
    if "verdict" not in data:
        print(f"run {run}: server said -> {data}")
        continue
    verdict = data["verdict"]
    print(f"run {run}: winner={verdict['winner']} shuffled={data['judge_saw_swapped']} "
          f"A={data['words_a']}w B={data['words_b']}w")
    print(f"   reason: {verdict['reason']}")
    for score in verdict["scores"]:
        print(f"   {score['label']}: {score['total']}/100 — {score['comment']}")
    leak = [w for w in ("Output A", "Output B", " A ", " B ") if w in verdict["reason"]]
    print(f"   label leak in reason: {leak or 'none'}\n")
