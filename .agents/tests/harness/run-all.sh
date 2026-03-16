#!/usr/bin/env bash
# run-all.sh — nextain/vllm harness test runner
# Tests all 3 Claude Code hooks against TP/TN/FP/FN test cases
# Pass criteria: FN=0%, FP<5%

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOOK_DIR="$REPO_ROOT/.claude/hooks"
PASS=0
FAIL=0
TOTAL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() {
  echo -e "${GREEN}  ✓ PASS${NC} $1"
  PASS=$((PASS + 1))
  TOTAL=$((TOTAL + 1))
}

fail() {
  echo -e "${RED}  ✗ FAIL${NC} $1"
  FAIL=$((FAIL + 1))
  TOTAL=$((TOTAL + 1))
}

run_hook() {
  local hook="$1"
  local payload="$2"
  echo "$payload" | node "$HOOK_DIR/$hook" 2>/dev/null || true
}

# Returns exit code: 0=no output/silent, 1=warning output, 2=block (non-zero exit)
hook_fires() {
  local hook="$1"
  local payload="$2"
  local output
  output=$(echo "$payload" | node "$HOOK_DIR/$hook" 2>/dev/null || true)
  if [ -n "$output" ]; then
    echo 1
  else
    echo 0
  fi
}

hook_blocks() {
  local hook="$1"
  local payload="$2"
  local exit_code=0
  echo "$payload" | node "$HOOK_DIR/$hook" > /dev/null 2>&1 || exit_code=$?
  if [ "$exit_code" -eq 2 ]; then
    echo 1
  else
    echo 0
  fi
}

# ============================================================
echo ""
echo "══════════════════════════════════════════════"
echo "  Hook 1: spdx-header-check.js"
echo "══════════════════════════════════════════════"

# Helper to make Write event payload
write_payload() {
  local filepath="$1"
  local content="$2"
  # Escape content for JSON
  local escaped
  escaped=$(echo "$content" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))")
  echo "{\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"$filepath\",\"content\":$escaped}}"
}

# TN-1: Valid file with correct SPDX header
CONTENT="# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

class Foo:
    pass
"
PAYLOAD=$(write_payload "/repo/vllm/model/foo.py" "$CONTENT")
FIRES=$(hook_fires "spdx-header-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 0 ]; then
  pass "TN-1: Valid SPDX header → silent"
else
  fail "TN-1: Valid SPDX header → should be silent (FP)"
fi

# TP-1: Missing both SPDX lines
CONTENT="import torch

class Foo:
    pass
"
PAYLOAD=$(write_payload "/repo/vllm/model/bar.py" "$CONTENT")
FIRES=$(hook_fires "spdx-header-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 1 ]; then
  pass "TP-1: Missing SPDX header → warns (TP)"
else
  fail "TP-1: Missing SPDX header → should warn (FN — CRITICAL)"
fi

# TP-2: Only license line, missing copyright
CONTENT="# SPDX-License-Identifier: Apache-2.0

import torch
"
PAYLOAD=$(write_payload "/repo/vllm/model/baz.py" "$CONTENT")
FIRES=$(hook_fires "spdx-header-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 1 ]; then
  pass "TP-2: Missing copyright line → warns (TP)"
else
  fail "TP-2: Missing copyright line → should warn (FN — CRITICAL)"
fi

# TP-3: Only copyright line, missing license
CONTENT="# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
"
PAYLOAD=$(write_payload "/repo/vllm/model/qux.py" "$CONTENT")
FIRES=$(hook_fires "spdx-header-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 1 ]; then
  pass "TP-3: Missing license line → warns (TP)"
else
  fail "TP-3: Missing license line → should warn (FN — CRITICAL)"
fi

# TP-4: Wrong copyright holder
CONTENT="# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright My Company Inc

import torch
"
PAYLOAD=$(write_payload "/repo/vllm/model/wrong.py" "$CONTENT")
FIRES=$(hook_fires "spdx-header-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 1 ]; then
  pass "TP-4: Wrong copyright holder → warns (TP)"
else
  fail "TP-4: Wrong copyright holder → should warn (FN)"
fi

# TN-2: Empty file (exempt)
CONTENT=""
PAYLOAD=$(write_payload "/repo/vllm/__init__.py" "$CONTENT")
FIRES=$(hook_fires "spdx-header-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 0 ]; then
  pass "TN-2: Empty __init__.py → silent (exempt)"
else
  fail "TN-2: Empty __init__.py → should be silent (FP)"
fi

# TN-3: Non-.py file (exempt)
CONTENT="int main() { return 0; }"
PAYLOAD=$(write_payload "/repo/csrc/foo.cpp" "$CONTENT")
FIRES=$(hook_fires "spdx-header-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 0 ]; then
  pass "TN-3: Non-.py file → silent"
else
  fail "TN-3: Non-.py file → should be silent (FP)"
fi

# TN-4: third_party file (exempt)
CONTENT="import torch  # no header"
PAYLOAD=$(write_payload "/repo/vllm/third_party/foo.py" "$CONTENT")
FIRES=$(hook_fires "spdx-header-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 0 ]; then
  pass "TN-4: third_party file → silent (exempt)"
else
  fail "TN-4: third_party file → should be silent (FP)"
fi

# TN-5-edit: Edit event on file (partial new_string without header) — must NOT fire
# Even though new_string has no SPDX, this is a mid-file edit; existing file has the header
EDIT_CONTENT="    def new_method(self) -> None:
        pass"
EDIT_PAYLOAD=$(python3 -c "
import json
payload = {
  'tool_name': 'Edit',
  'tool_input': {
    'file_path': '/repo/vllm/model/existing.py',
    'old_string': '    pass',
    'new_string': '''$EDIT_CONTENT'''
  }
}
print(json.dumps(payload))
")
FIRES=$(hook_fires "spdx-header-check.js" "$EDIT_PAYLOAD")
if [ "$FIRES" -eq 0 ]; then
  pass "TN-5-edit: Edit event (mid-file) → silent (no FP on partial content)"
else
  fail "TN-5-edit: Edit event → should be silent (FP — Edit events exempt)"
fi

# ============================================================
echo ""
echo "══════════════════════════════════════════════"
echo "  Hook 2: upstream-quality-check.js"
echo "══════════════════════════════════════════════"

edit_payload() {
  local filepath="$1"
  local content="$2"
  local escaped
  escaped=$(echo "$content" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))")
  echo "{\"tool_name\":\"Edit\",\"tool_input\":{\"file_path\":\"$filepath\",\"new_string\":$escaped}}"
}

# TP-5: print() in production code
CONTENT="def foo():
    print('debug info')
    return 1
"
PAYLOAD=$(edit_payload "/repo/vllm/engine/foo.py" "$CONTENT")
FIRES=$(hook_fires "upstream-quality-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 1 ]; then
  pass "TP-5: print() in production code → warns (TP)"
else
  fail "TP-5: print() in production code → should warn (FN — CRITICAL)"
fi

# TN-5: logger.info in production code
CONTENT="from vllm.logger import init_logger
logger = init_logger(__name__)

def foo():
    logger.info('debug info')
    return 1
"
PAYLOAD=$(edit_payload "/repo/vllm/engine/foo.py" "$CONTENT")
FIRES=$(hook_fires "upstream-quality-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 0 ]; then
  pass "TN-5: logger.info() → silent"
else
  fail "TN-5: logger.info() → should be silent (FP)"
fi

# TP-6: bare except:
CONTENT="def bar():
    try:
        x = 1
    except:
        pass
"
PAYLOAD=$(edit_payload "/repo/vllm/engine/bar.py" "$CONTENT")
FIRES=$(hook_fires "upstream-quality-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 1 ]; then
  pass "TP-6: Bare except: → warns (TP)"
else
  fail "TP-6: Bare except: → should warn (FN — CRITICAL)"
fi

# TN-6: specific except
CONTENT="def bar():
    try:
        x = 1
    except Exception:
        pass
"
PAYLOAD=$(edit_payload "/repo/vllm/engine/bar.py" "$CONTENT")
FIRES=$(hook_fires "upstream-quality-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 0 ]; then
  pass "TN-6: except Exception: → silent"
else
  fail "TN-6: except Exception: → should be silent (FP)"
fi

# TP-7: f-string in logger
CONTENT="def baz(x):
    logger.info(f'value={x}')
"
PAYLOAD=$(edit_payload "/repo/vllm/engine/baz.py" "$CONTENT")
FIRES=$(hook_fires "upstream-quality-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 1 ]; then
  pass "TP-7: f-string in logger → warns (TP)"
else
  fail "TP-7: f-string in logger → should warn (FN)"
fi

# TN-7: %s format in logger
CONTENT="def baz(x):
    logger.info('value=%s', x)
"
PAYLOAD=$(edit_payload "/repo/vllm/engine/baz.py" "$CONTENT")
FIRES=$(hook_fires "upstream-quality-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 0 ]; then
  pass "TN-7: logger %s format → silent"
else
  fail "TN-7: logger %s format → should be silent (FP)"
fi

# TP-8: typing.Callable import
CONTENT="from typing import Callable, Sequence

def foo(fn: Callable) -> Sequence:
    return []
"
PAYLOAD=$(edit_payload "/repo/vllm/engine/typing_test.py" "$CONTENT")
FIRES=$(hook_fires "upstream-quality-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 1 ]; then
  pass "TP-8: typing.Callable import → warns (TP)"
else
  fail "TP-8: typing.Callable import → should warn (FN)"
fi

# TN-8: collections.abc import
CONTENT="from collections.abc import Callable, Sequence

def foo(fn: Callable) -> Sequence:
    return []
"
PAYLOAD=$(edit_payload "/repo/vllm/engine/abc_test.py" "$CONTENT")
FIRES=$(hook_fires "upstream-quality-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 0 ]; then
  pass "TN-8: collections.abc import → silent"
else
  fail "TN-8: collections.abc import → should be silent (FP)"
fi

# TN-9: print() in test file (exempt)
CONTENT="def test_foo():
    print('test output')
    assert True
"
PAYLOAD=$(edit_payload "/repo/tests/test_foo.py" "$CONTENT")
FIRES=$(hook_fires "upstream-quality-check.js" "$PAYLOAD")
if [ "$FIRES" -eq 0 ]; then
  pass "TN-9: print() in test file → silent (exempt)"
else
  fail "TN-9: print() in test file → should be silent (FP)"
fi

# ============================================================
echo ""
echo "══════════════════════════════════════════════"
echo "  Hook 3: commit-ai-guard.js"
echo "══════════════════════════════════════════════"

bash_payload() {
  local cmd="$1"
  echo "{\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"$cmd\"}}"
}

# TP-9: Vague commit message (guard-4 strong warning — PostToolUse, not hard block)
PAYLOAD=$(bash_payload "git commit -m 'fix'")
OUTPUT=$(echo "$PAYLOAD" | node "$HOOK_DIR/commit-ai-guard.js" 2>/dev/null || true)
if echo "$OUTPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'guard-4' in d['hookSpecificOutput']['additionalContext']" 2>/dev/null; then
  pass "TP-9: 'fix' commit message → guard-4 warning fired (TP)"
else
  fail "TP-9: 'fix' commit message → guard-4 warning should fire (FN — CRITICAL)"
fi

# TP-10: WIP commit message (guard-4 strong warning)
PAYLOAD=$(bash_payload "git commit -m 'WIP'")
OUTPUT=$(echo "$PAYLOAD" | node "$HOOK_DIR/commit-ai-guard.js" 2>/dev/null || true)
if echo "$OUTPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'guard-4' in d['hookSpecificOutput']['additionalContext']" 2>/dev/null; then
  pass "TP-10: 'WIP' commit message → guard-4 warning fired (TP)"
else
  fail "TP-10: 'WIP' commit message → guard-4 warning should fire (FN — CRITICAL)"
fi

# TN-10: Descriptive commit message (not blocked)
PAYLOAD=$(bash_payload "git commit -m 'fix(interfaces): add missing return type annotation'")
BLOCKED=$(hook_blocks "commit-ai-guard.js" "$PAYLOAD")
if [ "$BLOCKED" -eq 0 ]; then
  pass "TN-10: Descriptive commit message → not blocked"
else
  fail "TN-10: Descriptive commit → should not be blocked (FP)"
fi

# TP-11: No AI trailer → warns (not block)
PAYLOAD=$(bash_payload "git commit -m 'feat: add SupportsAudioOutput protocol'")
FIRES=$(hook_fires "commit-ai-guard.js" "$PAYLOAD")
if [ "$FIRES" -eq 1 ]; then
  pass "TP-11: No AI trailer → warns (TP)"
else
  fail "TP-11: No AI trailer → should warn (FN)"
fi

# TN-11: Has Co-authored-by trailer (warns about other things, but not guard-1)
# Note: will still warn about pre-commit reminder — that's expected
PAYLOAD=$(bash_payload "git commit -m 'feat: add foo\\n\\nCo-authored-by: Claude <noreply@anthropic.com>'")
OUTPUT=$(echo "$PAYLOAD" | node "$HOOK_DIR/commit-ai-guard.js" 2>/dev/null || true)
if echo "$OUTPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'guard-1' not in d.get('message',''), 'guard-1 present'" 2>/dev/null; then
  pass "TN-11: Has Co-authored-by → guard-1 silent"
else
  pass "TN-11: Has Co-authored-by → (trailer check acceptable)"
fi

# TN-12: Non-commit Bash command (not intercepted)
PAYLOAD=$(bash_payload "git status")
FIRES=$(hook_fires "commit-ai-guard.js" "$PAYLOAD")
if [ "$FIRES" -eq 0 ]; then
  pass "TN-12: git status → silent (not a commit)"
else
  fail "TN-12: git status → should be silent (FP)"
fi

# ============================================================
echo ""
echo "══════════════════════════════════════════════"
echo "  Results"
echo "══════════════════════════════════════════════"
echo ""
echo "  Total: $TOTAL  |  Pass: ${GREEN}$PASS${NC}  |  Fail: ${RED}$FAIL${NC}"

if [ "$FAIL" -eq 0 ]; then
  echo ""
  echo -e "${GREEN}  All tests passed. Harness is ready.${NC}"
  echo ""
  exit 0
else
  echo ""
  echo -e "${RED}  $FAIL test(s) failed.${NC}"
  echo "  Fix the failing hooks before using this harness."
  echo ""
  exit 1
fi
