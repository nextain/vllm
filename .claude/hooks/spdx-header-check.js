#!/usr/bin/env node
/**
 * spdx-header-check.js
 * Trigger: Write | Edit on *.py files
 * Purpose: Enforce SPDX license header on all vLLM Python files
 *
 * Required header (exact):
 *   # SPDX-License-Identifier: Apache-2.0
 *   # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

import { readFileSync } from "fs";

const LICENSE_LINE =
  "# SPDX-License-Identifier: Apache-2.0";
const COPYRIGHT_LINE =
  "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project";

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

  // Only check .py files
  if (!filePath.endsWith(".py")) {
    process.exit(0);
  }

  // Only check Write events — Edit events modify existing files that already have the header.
  // Checking new_string (partial replacement) would cause false positives on mid-file edits.
  const toolName = event.tool_name || "";
  if (toolName !== "Write") {
    process.exit(0);
  }

  // Only check files inside vllm/ or tests/ directories
  const inScope =
    filePath.includes("/vllm/") ||
    filePath.includes("/tests/") ||
    /\/vllm-[^/]*\/vllm\//.test(filePath);
  if (!inScope) {
    process.exit(0);
  }

  // Skip third_party
  if (filePath.includes("/third_party/")) {
    process.exit(0);
  }

  // Read the full file content from Write event
  const newContent = event.tool_input?.content || "";

  // Empty file (e.g. empty __init__.py) — exempt
  if (!newContent || newContent.trim() === "") {
    process.exit(0);
  }

  const lines = newContent.split("\n");
  // Skip shebang line if present
  let startIdx = 0;
  if (lines[0] && lines[0].startsWith("#!")) {
    startIdx = 1;
  }

  const firstLines = lines.slice(startIdx, startIdx + 5).join("\n");

  const hasLicense = firstLines.includes(LICENSE_LINE);
  const hasCopyright = firstLines.includes(COPYRIGHT_LINE);

  if (!hasLicense || !hasCopyright) {
    const missing = [];
    if (!hasLicense) missing.push("SPDX-License-Identifier");
    if (!hasCopyright) missing.push("SPDX-FileCopyrightText");

    const message = [
      `⚠️  SPDX header missing in: ${filePath}`,
      `Missing: ${missing.join(", ")}`,
      ``,
      `Every vLLM Python file must start with:`,
      `  # SPDX-License-Identifier: Apache-2.0`,
      `  # SPDX-FileCopyrightText: Copyright contributors to the vLLM project`,
      ``,
      `This is enforced by upstream pre-commit (check-spdx-header hook).`,
      `Fix before committing — upstream PR will be rejected without it.`,
    ].join("\n");

    // Output as JSON for Claude Code hook response
    console.log(
      JSON.stringify({
        type: "warning",
        message,
      })
    );
    process.exit(0);
  }

  process.exit(0);
}

main();
