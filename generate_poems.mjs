/**
 * Generate poems using different models via @github/copilot-sdk.
 * Every model receives the same brief so you can compare their styles.
 * All requests run in parallel for speed.
 */

import { CopilotClient } from "@github/copilot-sdk";
import { writeFileSync } from "fs";

const SHARED_PROMPT =
  process.env.POEM_BRIEF || "Write a sonnet about the Forth Bridge.";

const MODELS = [
  ["gpt-5", "GPT-5"],
  ["gpt-4.1", "GPT-4.1"],
  ["gpt-4o-mini", "GPT-4o Mini"],
  ["claude-sonnet-4.5", "Claude Sonnet 4.5"],
  ["o3-mini", "o3-mini"],
  ["gpt-4o", "GPT-4o"],
  ["gemini-2.5-pro", "Gemini 2.5 Pro"],
];

const SYSTEM_MSG =
  "You are a talented poet. Respond ONLY with the poem itself — " +
  "no titles, no explanations, no extra commentary.";

// ── Helpers ────────────────────────────────────────────────

function separator() {
  return "=".repeat(60);
}

function extractVote(verdict, prefix) {
  for (const line of verdict.split("\n")) {
    if (line.toLowerCase().startsWith(prefix)) {
      const match = line.split(":")[1]?.match(/\d+/);
      if (match) return parseInt(match[0], 10);
    }
  }
  return null;
}

function extractReason(verdict, prefix) {
  for (const line of verdict.split("\n")) {
    if (line.toLowerCase().startsWith(prefix)) {
      const parts = line.split(":");
      if (parts.length > 1) return parts.slice(1).join(":").trim();
    }
  }
  return "";
}

// ── Main ───────────────────────────────────────────────────

const client = new CopilotClient();
await client.start();

// Phase 1: Generate poems
console.log(`Brief given to every model:\n"${SHARED_PROMPT}"\n`);
console.log("Calling all models in parallel…\n");

const results = {};

async function generatePoem(client, idx, model, label) {
  const session = await client.createSession({
    model,
    systemMessage: { mode: "replace", content: SYSTEM_MSG },
    infiniteSessions: { enabled: false },
  });
  try {
    const response = await session.sendAndWait(
      { prompt: SHARED_PROMPT },
      60_000
    );
    const poem = response?.data?.content?.trim() || "[EMPTY RESPONSE]";
    return { idx, label, model, poem };
  } catch (err) {
    return { idx, label, model, poem: `[ERROR] ${err.message}` };
  } finally {
    await session.destroy().catch(() => {});
  }
}

const poemPromises = MODELS.map(([model, label], i) =>
  generatePoem(client, i + 1, model, label).then((result) => {
    results[result.idx] = result;
    console.log(separator());
    console.log(`  Poem ${result.idx}: ${result.label}  (${result.model})`);
    console.log(separator() + "\n");
    console.log(result.poem);
    console.log();
    return result;
  })
);

await Promise.allSettled(poemPromises);

console.log(separator());
console.log(`  Done — ${MODELS.length} poems generated!`);
console.log(separator() + "\n");

// ── Phase 2: Voting ─────────────────────────────────────────

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function buildBallot(shuffledOrder) {
  const lines = [];
  const positionToOriginal = {};
  shuffledOrder.forEach((origIdx, i) => {
    const pos = i + 1;
    const { poem } = results[origIdx];
    lines.push(`--- Poem ${pos} ---\n${poem}\n`);
    positionToOriginal[pos] = origIdx;
  });
  return { ballot: lines.join("\n"), positionToOriginal };
}

console.log("\n" + separator());
console.log("  VOTING ROUND — each model picks the best AND worst poem");
console.log("  (poems shown in randomized order to each voter)");
console.log(separator() + "\n");
console.log(`Calling all ${MODELS.length} models in parallel…\n`);

const votes = {};

async function castVote(client, idx, model, label) {
  const poemIndices = shuffle(Object.keys(results).map(Number));
  const { ballot, positionToOriginal } = buildBallot(poemIndices);

  const votePrompt =
    `You are a poetry critic. Below are ${MODELS.length} poems written in response to the same brief.\n\n` +
    `Brief: "${SHARED_PROMPT}"\n\n` +
    `${ballot}\n` +
    "Vote for the BEST poem and the WORST poem.\n" +
    "Reply in this exact format:\n" +
    "Best: <poem number>\n" +
    "Best reason: <1–2 sentence explanation>\n" +
    "Worst: <poem number>\n" +
    "Worst reason: <1–2 sentence explanation>\n";

  const session = await client.createSession({
    model,
    systemMessage: { mode: "replace", content: "You are a poetry critic." },
    infiniteSessions: { enabled: false },
  });
  try {
    const response = await session.sendAndWait(
      { prompt: votePrompt },
      60_000
    );
    const verdict =
      response?.data?.content?.trim() || "[EMPTY RESPONSE]";
    return { idx, label, verdict, positionToOriginal };
  } catch (err) {
    return {
      idx,
      label,
      verdict: `[ERROR] ${err.message}`,
      positionToOriginal,
    };
  } finally {
    await session.destroy().catch(() => {});
  }
}

const votePromises = MODELS.map(([model, label], i) =>
  castVote(client, i + 1, model, label).then((result) => {
    votes[result.idx] = result;
    console.log(`  ${result.label}:`);
    for (const line of result.verdict.split("\n")) {
      console.log(`    ${line}`);
    }
    console.log();
    return result;
  })
);

await Promise.allSettled(votePromises);

// ── Tally ───────────────────────────────────────────────────

const bestTally = {};
const worstTally = {};
const bestComments = {};
const worstComments = {};

for (const idx of Object.keys(results)) {
  bestTally[idx] = 0;
  worstTally[idx] = 0;
  bestComments[idx] = [];
  worstComments[idx] = [];
}

for (const { label, verdict, positionToOriginal } of Object.values(votes)) {
  const b = extractVote(verdict, "best:");
  const w = extractVote(verdict, "worst:");
  const bestReason = extractReason(verdict, "best reason:");
  const worstReason = extractReason(verdict, "worst reason:");

  if (b && positionToOriginal) {
    const origB = positionToOriginal[b] ?? b;
    bestTally[origB] = (bestTally[origB] || 0) + 1;
    (bestComments[origB] ??= []).push([label, bestReason]);
  }
  if (w && positionToOriginal) {
    const origW = positionToOriginal[w] ?? w;
    worstTally[origW] = (worstTally[origW] || 0) + 1;
    (worstComments[origW] ??= []).push([label, worstReason]);
  }
}

const sortedByBest = Object.keys(results)
  .map(Number)
  .sort((a, b) => (bestTally[b] || 0) - (bestTally[a] || 0));
const sortedByWorst = Object.keys(results)
  .map(Number)
  .sort((a, b) => (worstTally[b] || 0) - (worstTally[a] || 0));

console.log("-".repeat(60));
console.log("  BEST POEM TALLY");
console.log("-".repeat(60));
for (const num of sortedByBest) {
  const { label } = results[num];
  console.log(`    Poem ${num} (${label}): ${bestTally[num] || 0} vote(s)`);
  for (const [voter, reason] of bestComments[num] || []) {
    console.log(`      - ${voter}: ${reason}`);
  }
}

console.log();
console.log("-".repeat(60));
console.log("  WORST POEM TALLY");
console.log("-".repeat(60));
for (const num of sortedByWorst) {
  const { label } = results[num];
  console.log(`    Poem ${num} (${label}): ${worstTally[num] || 0} vote(s)`);
  for (const [voter, reason] of worstComments[num] || []) {
    console.log(`      - ${voter}: ${reason}`);
  }
}

const winner = sortedByBest[0];
const loser = sortedByWorst[0];
console.log(`\n  🏆 Best:  Poem ${winner} — ${results[winner].label}`);
console.log(`  💀 Worst: Poem ${loser} — ${results[loser].label}`);
console.log(separator() + "\n");

// ── Write results.md ────────────────────────────────────────

const md = [];
md.push("# Poetry Competition Results", "");
md.push(`**Brief:** ${SHARED_PROMPT}`, "", "---", "", "## Poems", "");

for (const i of Object.keys(results).map(Number).sort()) {
  const { label, model, poem } = results[i];
  md.push(`### Poem ${i}: ${label}`);
  md.push(`*Model: \`${model}\`*`, "");
  md.push(poem, "");
}

md.push("---", "", "## Votes", "");

for (const i of Object.keys(votes).map(Number).sort()) {
  const { label, verdict } = votes[i];
  md.push(`### ${label}`, "");
  for (const line of verdict.split("\n")) {
    md.push(`> ${line}`);
  }
  md.push("");
}

md.push("---", "", "## Results", "", "### Best Poem Tally", "");
md.push("| Poem | Model | Votes |", "|------|-------|-------|");
for (const num of sortedByBest) {
  md.push(
    `| Poem ${num} | ${results[num].label} | ${bestTally[num] || 0} |`
  );
}

md.push("", "#### Comments on Best Poems", "");
for (const num of sortedByBest) {
  if (bestComments[num]?.length) {
    md.push(`**Poem ${num} (${results[num].label}):**`);
    for (const [voter, reason] of bestComments[num]) {
      md.push(`- *${voter}:* ${reason}`);
    }
    md.push("");
  }
}

md.push("### Worst Poem Tally", "");
md.push("| Poem | Model | Votes |", "|------|-------|-------|");
for (const num of sortedByWorst) {
  md.push(
    `| Poem ${num} | ${results[num].label} | ${worstTally[num] || 0} |`
  );
}

md.push("", "#### Comments on Worst Poems", "");
for (const num of sortedByWorst) {
  if (worstComments[num]?.length) {
    md.push(`**Poem ${num} (${results[num].label}):**`);
    for (const [voter, reason] of worstComments[num]) {
      md.push(`- *${voter}:* ${reason}`);
    }
    md.push("");
  }
}

md.push("", "---", "");
md.push(`🏆 **Best:** Poem ${winner} — ${results[winner].label}`, "");
md.push(`💀 **Worst:** Poem ${loser} — ${results[loser].label}`, "");

writeFileSync("results.md", md.join("\n"));
console.log("Results written to results.md");

await client.stop();
