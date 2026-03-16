#!/usr/bin/env node
/**
 * commit-ai-guard.js
 * Trigger: Bash containing 'git commit'
 * Purpose: Enforce AI-Native commit quality before code leaves the editor
 *
 * Checks:
 *   guard-1: Warn if Co-authored-by AI trailer appears missing
 *   guard-2: Remind to run duplicate PR check before upstream PR
 *   guard-3: Remind to run pre-commit
 *   guard-4: BLOCK vague single-word commit messages (fix, WIP, temp, test)
 */

import { readFileSync } from "fs";

const VAGUE_MESSAGE_REGEX = /^(WIP|wip|fix|Fix|FIX|temp|Temp|TEMP|test|Test|update|Update|UPDATE|wip:|todo|TODO)\s*$/;

// Messages that suggest AI was used (co-authored, assisted, etc.)
const AI_TRAILER_REGEX = /Co-authored-by:|Assisted-by:|Generated-by:/i;

function main() {
  let input = "";
  try {
    input = readFileSync("/dev/stdin", "utf8");
  } catch {
    process.exit(0);
  }

  let event;
  try {
    event = JSON.parse(input);
  } catch {
    process.exit(0);
  }

  // Only intercept git commit commands
  const command = event.tool_input?.command || "";
  if (!command.includes("git commit")) {
    process.exit(0);
  }

  // Skip --amend, --no-edit (already committed, skip guard)
  if (command.includes("--no-edit")) {
    process.exit(0);
  }

  // Extract commit message from -m flag if present
  const mMatch = command.match(/-m\s+['"]([^'"]*)['"]/s) ||
    command.match(/-m\s+"((?:[^"\\]|\\.)*)"/s) ||
    command.match(/-m\s+'((?:[^'\\]|\\.)*)'/s);
  const commitMessage = mMatch ? mMatch[1] : null;

  const warnings = [];
  const blocks = [];

  // guard-4: Block vague single-word messages
  if (commitMessage) {
    const firstLine = commitMessage.split("\n")[0].trim();
    if (VAGUE_MESSAGE_REGEX.test(firstLine)) {
      blocks.push(
        `guard-4: Vague commit message blocked: "${firstLine}"\n` +
        `  Use a descriptive message, e.g.:\n` +
        `    fix(interfaces): add missing return type annotation\n` +
        `    feat: add SupportsAudioOutput protocol interface`
      );
    }
  }

  // guard-1: Warn if no AI trailer (when committing Python files)
  if (!commitMessage || !AI_TRAILER_REGEX.test(commitMessage)) {
    warnings.push(
      `guard-1: No AI attribution trailer found.\n` +
      `  If Claude assisted with this commit, add:\n` +
      `    Co-authored-by: Claude <noreply@anthropic.com>\n` +
      `  (Required for upstream vLLM PR — see AGENTS.md §1.4)`
    );
  }

  // guard-2: Remind about duplicate PR check
  warnings.push(
    `guard-2: Upstream PR reminder — did you run the duplicate check?\n` +
    `  gh pr list --repo vllm-project/vllm --state open --search "<keywords>"\n` +
    `  (Required before opening any upstream PR per AGENTS.md §1.1)`
  );

  // guard-3: Remind about pre-commit
  warnings.push(
    `guard-3: Did you run pre-commit?\n` +
    `  pre-commit run\n` +
    `  (Checks SPDX header, ruff, mypy, Signed-off-by, and more)`
  );

  // Output result
  if (blocks.length > 0) {
    // BLOCK: output to stderr and exit non-zero
    const message = [
      `🚫 Commit blocked by commit-ai-guard:`,
      ``,
      ...blocks,
    ].join("\n");

    console.log(JSON.stringify({
      type: "block",
      message,
    }));
    process.exit(2);
  }

  if (warnings.length > 0) {
    const message = [
      `📋 Pre-commit checklist (nextain/vllm upstream quality):`,
      ``,
      ...warnings.map((w, i) => `${i + 1}. ${w}`),
      ``,
      `These are reminders — the commit will proceed.`,
      `Fix Co-authored-by and pre-commit issues if targeting upstream PR.`,
    ].join("\n");

    console.log(JSON.stringify({
      type: "warning",
      message,
    }));
  }

  process.exit(0);
}

main();
