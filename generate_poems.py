"""
Generate 6 poems using 6 different models from GitHub Models.
Every model receives the same brief so you can compare their styles.
All API calls run in parallel for speed.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

TOKEN = os.environ["GITHUB_TOKEN"]
ENDPOINT = "https://models.github.ai/inference"

DEFAULT_PROMPT = "Write a poem about the Forth Bridge."

SHARED_PROMPT = os.environ.get("POEM_BRIEF") or DEFAULT_PROMPT

MODELS = [
    ("openai/gpt-5",                    "GPT-5"),
    ("openai/gpt-4.1",                  "GPT-4.1"),
    ("openai/gpt-4o-mini",              "GPT-4o Mini"),
    ("meta/Meta-Llama-3.1-8B-Instruct", "Llama 3.1 8B"),
    ("meta/Llama-3.3-70B-Instruct",     "Llama 3.3 70B"),
    ("openai/gpt-4o",                   "GPT-4o"),
    ("Cohere-command-r-plus-08-2024",   "Cohere Command R+"),
]

SYSTEM_MSG = (
    "You are a talented poet. Respond ONLY with the poem itself — "
    "no titles, no explanations, no extra commentary."
)

client = OpenAI(base_url=ENDPOINT, api_key=TOKEN, timeout=30.0)


# Models that require max_completion_tokens instead of max_tokens
NEW_STYLE_MODELS = {"openai/gpt-5", "openai/o3", "openai/o3-mini", "openai/o4-mini", "openai/o1", "openai/o1-mini"}


def generate_poem(model: str) -> str:
    """Call a single model and return the poem text."""
    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": SHARED_PROMPT},
        ],
    )
    if model in NEW_STYLE_MODELS:
        kwargs["max_completion_tokens"] = 2000
    else:
        kwargs["temperature"] = 0.9
        kwargs["max_tokens"] = 400
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


print(f"Brief given to every model:\n\"{SHARED_PROMPT}\"\n")
print("Calling all models in parallel…\n")

# Launch all requests at once and print as they complete
results: dict[int, str] = {}
with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
    futures = {
        pool.submit(generate_poem, model): (i, label)
        for i, (model, label) in enumerate(MODELS, 1)
    }
    for future in as_completed(futures):
        idx, label = futures[future]
        model = MODELS[idx - 1][0]
        try:
            poem = future.result()
            results[idx] = (label, model, poem)
        except Exception as exc:
            poem = f"[ERROR] {exc}"
            results[idx] = (label, model, poem)
        # Print immediately as each poem arrives
        print(f"{'='*60}")
        print(f"  Poem {idx}: {label}  ({model})")
        print(f"{'='*60}\n")
        print(poem)
        print()

print(f"{'='*60}")
print("  Done — 5 poems generated with 5 different GitHub Models!")
print(f"{'='*60}\n")

# ── Phase 2: Voting ─────────────────────────────────────────
# Build a ballot listing all poems (anonymised as Poem 1–5)
ballot_lines = []
for i in sorted(results):
    label, model, poem = results[i]
    ballot_lines.append(f"--- Poem {i} ---\n{poem}\n")
ballot = "\n".join(ballot_lines)

VOTE_PROMPT = (
    "You are a poetry critic. Below are 7 poems written in response to the same brief.\n\n"
    f"Brief: \"{SHARED_PROMPT}\"\n\n"
    f"{ballot}\n"
    "Vote for the BEST poem and the WORST poem.\n"
    "Reply in this exact format:\n"
    "Best: <poem number>\n"
    "Best reason: <1–2 sentence explanation>\n"
    "Worst: <poem number>\n"
    "Worst reason: <1–2 sentence explanation>\n"
)


def cast_vote(model: str, own_number: int) -> str:
    prompt = VOTE_PROMPT.replace("{own}", str(own_number))
    kwargs: dict = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    if model in NEW_STYLE_MODELS:
        kwargs["max_completion_tokens"] = 4000
    else:
        kwargs["temperature"] = 0.3
        kwargs["max_tokens"] = 250
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


print("\n" + "=" * 60)
print("  VOTING ROUND — each model picks the best AND worst poem")
print("=" * 60 + "\n")
print("Calling all 5 models in parallel…\n")

votes: dict[int, str] = {}
with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
    vote_futures = {
        pool.submit(cast_vote, model, i): (i, label)
        for i, (model, label) in enumerate(MODELS, 1)
    }
    for future in as_completed(vote_futures):
        idx, label = vote_futures[future]
        try:
            verdict = future.result()
            votes[idx] = (label, verdict)
        except Exception as exc:
            verdict = f"[ERROR] {exc}"
            votes[idx] = (label, verdict)
        # Print immediately as each vote arrives
        print(f"  {label}:")
        for line in verdict.splitlines():
            print(f"    {line}")
        print()

# Tally
from collections import Counter

def extract_vote(verdict: str, prefix: str) -> int | None:
    for line in verdict.splitlines():
        if line.lower().startswith(prefix):
            try:
                return int("".join(c for c in line.split(":")[1] if c.isdigit()))
            except ValueError:
                pass
    return None

def extract_reason(verdict: str, prefix: str) -> str:
    for line in verdict.splitlines():
        if line.lower().startswith(prefix):
            parts = line.split(":", 1)
            if len(parts) > 1:
                return parts[1].strip()
    return ""

best_tally: Counter[int] = Counter()
worst_tally: Counter[int] = Counter()
best_comments: dict[int, list[tuple[str, str]]] = {i: [] for i in results}
worst_comments: dict[int, list[tuple[str, str]]] = {i: [] for i in results}

for label, verdict in votes.values():
    b = extract_vote(verdict, "best:")
    w = extract_vote(verdict, "worst:")
    best_reason = extract_reason(verdict, "best reason:")
    worst_reason = extract_reason(verdict, "worst reason:")
    if b:
        best_tally[b] += 1
        best_comments[b].append((label, best_reason))
    if w:
        worst_tally[w] += 1
        worst_comments[w].append((label, worst_reason))

print("-" * 60)
print("  BEST POEM TALLY")
print("-" * 60)
for poem_num in sorted(results, key=lambda p: best_tally.get(p, 0), reverse=True):
    _, lbl, _ = results[poem_num]
    count = best_tally.get(poem_num, 0)
    print(f"    Poem {poem_num} ({lbl}): {count} vote(s)")
    for voter, reason in best_comments[poem_num]:
        print(f"      - {voter}: {reason}")

print()
print("-" * 60)
print("  WORST POEM TALLY")
print("-" * 60)
for poem_num in sorted(results, key=lambda p: worst_tally.get(p, 0), reverse=True):
    _, lbl, _ = results[poem_num]
    count = worst_tally.get(poem_num, 0)
    print(f"    Poem {poem_num} ({lbl}): {count} vote(s)")
    for voter, reason in worst_comments[poem_num]:
        print(f"      - {voter}: {reason}")

winner = best_tally.most_common(1)[0][0]
_, winner_label, _ = results[winner]
loser = worst_tally.most_common(1)[0][0]
_, loser_label, _ = results[loser]
print(f"\n  🏆 Best:  Poem {winner} — {winner_label}")
print(f"  💀 Worst: Poem {loser} — {loser_label}")
print("=" * 60 + "\n")

# ── Write results.md ────────────────────────────────────────
md_lines = [
    "# Poetry Competition Results",
    "",
    f"**Brief:** {SHARED_PROMPT}",
    "",
    "---",
    "",
    "## Poems",
    "",
]

for i in sorted(results):
    label, model, poem = results[i]
    md_lines.append(f"### Poem {i}: {label}")
    md_lines.append(f"*Model: `{model}`*")
    md_lines.append("")
    md_lines.append(poem)
    md_lines.append("")

md_lines.append("---")
md_lines.append("")
md_lines.append("## Votes")
md_lines.append("")

for i in sorted(votes):
    label, verdict = votes[i]
    md_lines.append(f"### {label}")
    md_lines.append("")
    for line in verdict.splitlines():
        md_lines.append(f"> {line}")
    md_lines.append("")

md_lines.append("---")
md_lines.append("")
md_lines.append("## Results")
md_lines.append("")
md_lines.append("### Best Poem Tally")
md_lines.append("")
md_lines.append("| Poem | Model | Votes |")
md_lines.append("|------|-------|-------|")
for poem_num in sorted(results, key=lambda p: best_tally.get(p, 0), reverse=True):
    _, lbl, _ = results[poem_num]
    count = best_tally.get(poem_num, 0)
    md_lines.append(f"| Poem {poem_num} | {lbl} | {count} |")

md_lines.append("")
md_lines.append("#### Comments on Best Poems")
md_lines.append("")
for poem_num in sorted(results, key=lambda p: best_tally.get(p, 0), reverse=True):
    if best_comments[poem_num]:
        _, lbl, _ = results[poem_num]
        md_lines.append(f"**Poem {poem_num} ({lbl}):**")
        for voter, reason in best_comments[poem_num]:
            md_lines.append(f"- *{voter}:* {reason}")
        md_lines.append("")

md_lines.append("### Worst Poem Tally")
md_lines.append("")
md_lines.append("| Poem | Model | Votes |")
md_lines.append("|------|-------|-------|")
for poem_num in sorted(results, key=lambda p: worst_tally.get(p, 0), reverse=True):
    _, lbl, _ = results[poem_num]
    count = worst_tally.get(poem_num, 0)
    md_lines.append(f"| Poem {poem_num} | {lbl} | {count} |")

md_lines.append("")
md_lines.append("#### Comments on Worst Poems")
md_lines.append("")
for poem_num in sorted(results, key=lambda p: worst_tally.get(p, 0), reverse=True):
    if worst_comments[poem_num]:
        _, lbl, _ = results[poem_num]
        md_lines.append(f"**Poem {poem_num} ({lbl}):**")
        for voter, reason in worst_comments[poem_num]:
            md_lines.append(f"- *{voter}:* {reason}")
        md_lines.append("")

md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append(f"🏆 **Best:** Poem {winner} — {winner_label}")
md_lines.append("")
md_lines.append(f"💀 **Worst:** Poem {loser} — {loser_label}")
md_lines.append("")

with open("results.md", "w") as f:
    f.write("\n".join(md_lines))

print("Results written to results.md")
