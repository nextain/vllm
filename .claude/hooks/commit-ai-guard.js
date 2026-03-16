#!/usr/bin/env node
/**
 * commit-ai-guard.js
 * Trigger: PostToolUse on Bash (git commit)
 * Purpose: Enforce AI-Native commit quality
 *
 * Checks:
 *   guard-1: Warn if Co-authored-by AI trailer missing
 *   guard-2: Remind to run duplicate PR check before upstream PR
 *   guard-3: Remind to run pre-commit
 *   guard-4: Strong warning on vague single-word commit messages (fix, WIP, temp, test)
 */

const VAGUE_MESSAGE_REGEX =
  /^(WIP|wip|fix|Fix|FIX|temp|Temp|TEMP|test|Test|update|Update|UPDATE|wip:|todo|TODO)\s*$/;
const AI_TRAILER_REGEX = /Co-authored-by:|Assisted-by:|Generated-by:/i;

async function main() {
  let input = "";
  for await (const chunk of process.stdin) {
    input += chunk;
  }

  let data;
  try {
    data = JSON.parse(input);
  } catch {
    process.exit(0);
  }

  const toolName = data.tool_name || "";
  const command = data.tool_input?.command || "";

  if (toolName !== "Bash") {
    process.exit(0);
  }

  if (!command.match(/git\s+commit\b/)) {
    process.exit(0);
  }

  // Skip --no-edit amends
  if (command.includes("--no-edit")) {
    process.exit(0);
  }

  // Extract commit message from -m flag (simple string match, not heredoc)
  const mMatch =
    command.match(/-m\s+"((?:[^"\\]|\\.)*)"/s) ||
    command.match(/-m\s+'((?:[^'\\]|\\.)*)'/s);
  const commitMessage = mMatch ? mMatch[1] : null;

  const notices = [];

  // guard-4: Warn strongly on vague message
  if (commitMessage) {
    const firstLine = commitMessage.split("\n")[0].trim();
    if (VAGUE_MESSAGE_REGEX.test(firstLine)) {
      notices.push(
        `[guard-4] 🚫 Vague commit message: "${firstLine}"\n` +
        `  Upstream vLLM reviewers will reject PRs with non-descriptive history.\n` +
        `  Use: fix(interfaces): add missing return type annotation`
      );
    }
  }

  // guard-1: Warn if no AI trailer
  if (!commitMessage || !AI_TRAILER_REGEX.test(commitMessage)) {
    notices.push(
      `[guard-1] If Claude assisted, add trailer:\n` +
      `  Co-authored-by: Claude <noreply@anthropic.com>\n` +
      `  (Required for upstream vLLM PR — AGENTS.md §1.4)`
    );
  }

  // guard-2: Duplicate PR reminder
  notices.push(
    `[guard-2] Before opening upstream PR, run duplicate check:\n` +
    `  gh pr list --repo vllm-project/vllm --state open --search "<keywords>"`
  );

  // guard-3: pre-commit reminder
  notices.push(
    `[guard-3] Did you run pre-commit?\n` +
    `  pre-commit run  (checks SPDX, ruff, mypy, Signed-off-by)`
  );

  if (notices.length > 0) {
    const msg = [
      `📋 nextain/vllm commit checklist:`,
      ``,
      ...notices,
    ].join("\n");

    process.stdout.write(
      JSON.stringify({
        reason: "",
        hookSpecificOutput: {
          hookEventName: "PostToolUse",
          additionalContext: msg,
        },
      })
    );
  }

  process.exit(0);
}

main().catch(() => process.exit(0));
