#!/usr/bin/env node
/**
 * upstream-quality-check.js
 * Trigger: Write | Edit on vllm/**\/*.py
 * Purpose: Detect ruff-catchable anti-patterns before pre-commit runs
 *          Operates entirely with regex — no Python/ruff installation needed
 *
 * Checks:
 *   qual-1: print() in production code (should use logger)
 *   qual-2: bare except: (should be except Exception: or specific)
 *   qual-3: f-string in logger calls (should use %s format)
 *   qual-4: typing.Callable/Sequence direct import (use collections.abc)
 */

import { readFileSync } from "fs";

// Patterns: [id, regex, message, excludeInTests]
const CHECKS = [
  {
    id: "qual-1",
    // Match print( at start of line (with optional indentation), not in comments
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
    // Match bare except: with nothing after the colon (except whitespace/comment)
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
    // Match logger.*(f"..." or f'...')
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
    // Match direct import of Callable, Sequence, Mapping from typing (not collections.abc)
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

  const filePath = event.tool_input?.file_path || event.tool_input?.path || "";

  // Only check .py files inside vllm/ package (not docs, examples, etc.)
  if (!filePath.endsWith(".py")) {
    process.exit(0);
  }

  const inVllmPackage =
    /\/vllm\/[^/]/.test(filePath) ||
    filePath.includes("/vllm/model_executor/") ||
    filePath.includes("/vllm/entrypoints/") ||
    filePath.includes("/vllm/engine/");
  const inTests = filePath.includes("/tests/");

  if (!inVllmPackage && !inTests) {
    process.exit(0);
  }

  if (filePath.includes("/third_party/")) {
    process.exit(0);
  }

  const content = event.tool_input?.content || event.tool_input?.new_string || "";
  if (!content) {
    process.exit(0);
  }

  const violations = [];

  for (const check of CHECKS) {
    // Skip print() check in test files
    if (check.excludeTests && inTests) {
      continue;
    }

    if (check.regex.test(content)) {
      // Find the line number for context
      const lines = content.split("\n");
      let lineNum = -1;
      for (let i = 0; i < lines.length; i++) {
        if (check.regex.test(lines[i])) {
          lineNum = i + 1;
          break;
        }
      }
      violations.push(
        `[${check.id}] Line ~${lineNum}: ${check.message}`
      );
    }
  }

  if (violations.length > 0) {
    const message = [
      `⚠️  Upstream quality issues in: ${filePath}`,
      ``,
      violations.join("\n\n"),
      ``,
      `Fix these before running pre-commit.`,
      `These patterns will fail upstream CI and trigger PR rejection.`,
    ].join("\n");

    console.log(JSON.stringify({ type: "warning", message }));
  }

  process.exit(0);
}

main();
