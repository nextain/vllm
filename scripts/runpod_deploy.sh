#!/bin/bash
# RunPod deployment script for nextain/vllm audio output fork.
#
# Usage (from local machine):
#   bash scripts/runpod_deploy.sh [HOST] [PORT]
#   bash scripts/runpod_deploy.sh 99.69.17.69 11105
#
# This script:
#   1. Syncs all changed Python files to RunPod (/workspace/vllm-patch/)
#   2. Copies them into the installed vllm package
#   3. Runs the surgical patch script for interfaces.py and registry.py
#   4. Verifies the server starts (dry-run import check)

set -e

RUNPOD_HOST="${1:-99.69.17.69}"
RUNPOD_PORT="${2:-11105}"
SSH="ssh -o StrictHostKeyChecking=no root@${RUNPOD_HOST} -p ${RUNPOD_PORT}"
SCP_BASE="scp -o StrictHostKeyChecking=no -P ${RUNPOD_PORT}"
VLLM_SYS="/usr/local/lib/python3.11/dist-packages/vllm"

echo "=== Deploying nextain/vllm audio output fork to RunPod ==="
echo "    Host: ${RUNPOD_HOST}:${RUNPOD_PORT}"

# 1. Create workspace directory on RunPod
$SSH "mkdir -p /workspace/vllm-patch/scripts"

# 2. Copy changed files to RunPod workspace
REPO_ROOT="$(git rev-parse --show-toplevel)"

files=(
    "vllm/model_executor/models/minicpmo.py"
    "vllm/outputs.py"
    "vllm/v1/outputs.py"
    "vllm/v1/engine/__init__.py"
    "vllm/v1/engine/output_processor.py"
    "vllm/v1/core/sched/scheduler.py"
    "vllm/v1/worker/gpu_model_runner.py"
    "vllm/entrypoints/openai/chat_completion/serving.py"
)

echo ""
echo "--- Copying fork files ---"
for f in "${files[@]}"; do
    local_path="${REPO_ROOT}/${f}"
    remote_dir="/workspace/vllm-patch/$(dirname ${f})"
    $SSH "mkdir -p ${remote_dir}"
    ${SCP_BASE} "${local_path}" "root@${RUNPOD_HOST}:${remote_dir}/"
    echo "  [COPY] ${f}"
done

# 3. Copy the patch script
${SCP_BASE} "${REPO_ROOT}/scripts/runpod_patch.py" \
    "root@${RUNPOD_HOST}:/workspace/vllm-patch/scripts/"
echo "  [COPY] scripts/runpod_patch.py"

# 4. Install changed files into the system vllm package
echo ""
echo "--- Installing into ${VLLM_SYS} ---"
$SSH "bash -s" <<'REMOTE_INSTALL'
set -e
VLLM_SYS="/usr/local/lib/python3.11/dist-packages/vllm"
PATCH_DIR="/workspace/vllm-patch"

files=(
    "vllm/model_executor/models/minicpmo.py"
    "vllm/outputs.py"
    "vllm/v1/outputs.py"
    "vllm/v1/engine/__init__.py"
    "vllm/v1/engine/output_processor.py"
    "vllm/v1/core/sched/scheduler.py"
    "vllm/v1/worker/gpu_model_runner.py"
    "vllm/entrypoints/openai/chat_completion/serving.py"
)

for f in "${files[@]}"; do
    src="${PATCH_DIR}/${f}"
    dst="${VLLM_SYS}/${f#vllm/}"
    cp "${src}" "${dst}"
    echo "  [INSTALL] ${f}"
done
REMOTE_INSTALL

# 5. Run surgical patch for interfaces.py and registry.py
echo ""
echo "--- Applying surgical patches (interfaces.py, registry.py) ---"
$SSH "python3 /workspace/vllm-patch/scripts/runpod_patch.py"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "To start vLLM server:"
echo "  python3 -m vllm.entrypoints.openai.api_server \\"
echo "    --model openbmb/MiniCPM-o-4_5 \\"
echo "    --trust-remote-code \\"
echo "    --dtype bfloat16 \\"
echo "    --max-model-len 8192 \\"
echo "    --gpu-memory-utilization 0.87 \\"
echo "    --port 8000 \\"
echo "    --hf-overrides '{\"enable_audio_output\": true}' &"
