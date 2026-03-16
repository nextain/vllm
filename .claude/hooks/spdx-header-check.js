#!/usr/bin/env node
/**
 * spdx-header-check.js
 * Trigger: PostToolUse on Write (*.py)
 * Purpose: Enforce SPDX license header on all vLLM Python files
 *
 * Required header (exact):
 *   # SPDX-License-Identifier: Apache-2.0
 *   # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

const LICENSE_LINE = "# SPDX-License-Identifier: Apache-2.0";
const COPYRIGHT_LINE =
  "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project";

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

  // Only Write events — Edit events modify existing files (partial new_string = FP risk)
  if (toolName !== "Write") {
    process.exit(0);
  }

  // Only .py files
  if (!filePath.endsWith(".py")) {
    process.exit(0);
  }

  // Only files inside vllm/ or tests/ directories
  const norm = filePath.replace(/\\/g, "/");
  const inScope = norm.includes("/vllm/") || norm.includes("/tests/");
  if (!inScope) {
    process.exit(0);
  }

  // Skip third_party
  if (norm.includes("/third_party/")) {
    process.exit(0);
  }

  const content = data.tool_input?.content || "";

  // Empty file (e.g. empty __init__.py) — exempt
  if (!content || content.trim() === "") {
    process.exit(0);
  }

  const lines = content.split("\n");
  // Skip shebang if present
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

    const msg = [
      `⚠️  SPDX header missing in: ${filePath}`,
      `Missing: ${missing.join(", ")}`,
      ``,
      `Every vLLM Python file must start with:`,
      `  # SPDX-License-Identifier: Apache-2.0`,
      `  # SPDX-FileCopyrightText: Copyright contributors to the vLLM project`,
      ``,
      `Enforced by upstream pre-commit (check-spdx-header).`,
      `Fix before committing — upstream PR will be rejected without it.`,
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
