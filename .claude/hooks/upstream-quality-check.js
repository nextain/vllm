#!/usr/bin/env node
/**
 * upstream-quality-check.js
 * Trigger: PostToolUse on Edit|Write (*.py in vllm/ or tests/)
 * Purpose: Detect ruff-catchable anti-patterns before pre-commit runs
 *          Regex-based — no Python/ruff installation needed
 *
 * Checks:
 *   qual-1: print() in production code (should use logger)
 *   qual-2: bare except: (should be except Exception: or specific)
 *   qual-3: f-string in logger calls (should use %s format, ruff G004)
 *   qual-4: typing.Callable/Sequence direct import (use collections.abc, ruff UP035)
 */

const CHECKS = [
  {
    id: "qual-1",
    regex: /^(?!#)[\t ]*print\s*\(/m,
    message: [
      "qual-1: print() found — use structured logger instead.",
      "  from vllm.logger import init_logger",
      "  logger = init_logger(__name__)",
      "  logger.info('message %s', value)",
      "Upstream vLLM rejects production code with print() calls.",
    ].join("\n"),
    excludeTests: true,
  },
  {
    id: "qual-2",
    regex: /^[\t ]*except\s*:\s*(#.*)?$/m,
    message: [
      "qual-2: Bare except: found — specify exception type.",
      "  except Exception:  # or a specific exception like except ValueError:",
      "Upstream vLLM rejects bare except: (ruff B001/E722).",
    ].join("\n"),
    excludeTests: false,
  },
  {
    id: "qual-3",
    regex: /logger\s*\.\s*(debug|info|warning|error|critical|exception)\s*\(\s*f['"]/m,
    message: [
      "qual-3: f-string in logger call — use %s format.",
      "  ❌ logger.info(f'value={x}')",
      "  ✅ logger.info('value=%s', x)",
      "Ruff G004 rule enforces this in CI.",
    ].join("\n"),
    excludeTests: false,
  },
  {
    id: "qual-4",
    regex: /from\s+typing\s+import\s+[^#\n]*\b(Callable|Sequence|Mapping|MutableSequence|AsyncGenerator|Iterator|Generator|Iterable)\b/m,
    message: [
      "qual-4: Direct typing.Callable/Sequence import — use collections.abc.",
      "  ❌ from typing import Callable, Sequence",
      "  ✅ from collections.abc import Callable, Sequence",
      "Ruff UP035 rule enforces this. Required for Python 3.10+ compatibility.",
    ].join("\n"),
    excludeTests: false,
  },
];

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
  const filePath = data.tool_input?.file_path || data.parameters?.file_path || "";

  if (toolName !== "Edit" && toolName !== "Write") {
    process.exit(0);
  }

  if (!filePath.endsWith(".py")) {
    process.exit(0);
  }

  const norm = filePath.replace(/\\/g, "/");
  const inVllmPackage = /\/vllm\/[^/]/.test(norm);
  const inTests = norm.includes("/tests/");

  if (!inVllmPackage && !inTests) {
    process.exit(0);
  }

  if (norm.includes("/third_party/")) {
    process.exit(0);
  }

  const content = data.tool_input?.content || data.tool_input?.new_string || "";
  if (!content) {
    process.exit(0);
  }

  const violations = [];

  for (const check of CHECKS) {
    if (check.excludeTests && inTests) continue;

    if (check.regex.test(content)) {
      const lines = content.split("\n");
      let lineNum = -1;
      for (let i = 0; i < lines.length; i++) {
        if (check.regex.test(lines[i])) {
          lineNum = i + 1;
          break;
        }
      }
      violations.push(`[${check.id}] Line ~${lineNum}: ${check.message}`);
    }
  }

  if (violations.length > 0) {
    const msg = [
      `⚠️  Upstream quality issues in: ${filePath}`,
      ``,
      violations.join("\n\n"),
      ``,
      `Fix before running pre-commit. These patterns fail upstream CI and trigger PR rejection.`,
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
