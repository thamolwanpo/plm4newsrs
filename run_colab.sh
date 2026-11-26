#!/usr/bin/env bash
# Provision a Google Colab runtime via google-colab-cli, mount Google Drive,
# clone/update this repo, install dependencies, and run training with DRIVE_ROOT.
#
# Prerequisites (local machine):
#   pip install google-colab-cli   # or: uv tool install google-colab-cli
#
# Expected Google Drive layout under DRIVE_ROOT (default: /content/drive/MyDrive/P2):
#   $DRIVE_ROOT/data/<dataset>/    (input data from fakeun-dataprep)
#   $DRIVE_ROOT/outputs/<dataset>/ (created automatically for checkpoints & results)
#   $DRIVE_ROOT/logs/              (created automatically for batch logs)
#
# Usage examples:
#   # Default CPU runtime, default DRIVE_ROOT (/content/drive/MyDrive/P2), runs ./train.sh
#   ./run_colab.sh
#
#   # Custom Drive root on CPU
#   DRIVE_ROOT="/content/drive/MyDrive/my_project" ./run_colab.sh
#
#   # Use a GPU runtime (e.g. T4, L4, A100)
#   GPU=T4 DRIVE_ROOT="/content/drive/MyDrive/P2" ./run_colab.sh
#
#   # Run oracle + poisoned script instead of train.sh
#   SCRIPT=./train_oracle_and_poisoned.sh DRIVE_ROOT="/content/drive/MyDrive/P2" ./run_colab.sh
#
#   # Forward train.sh options (MODE, POISON_SETS, FLAG_LEVELS, NUM_SEEDS, CONFIG, DATA_ROOT)
#   MODE=oracle POISON_SETS="A_r0.5_s1" ./run_colab.sh
#
#   # Reuse an existing Colab session without creating a new one
#   REUSE_SESSION=1 SESSION=plm4newsrs ./run_colab.sh
#
#   # Auto-stop the Colab runtime when the job finishes
#   STOP_ON_FINISH=1 ./run_colab.sh

set -euo pipefail

SESSION="${SESSION:-plm4newsrs}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/efficient_fake_news_unlearning}"
REPO_URL="${REPO_URL:-https://github.com/thamolwanpo/plm4newsrs.git}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/content/plm4newsrs}"
SCRIPT="${SCRIPT:-./train.sh}"
GPU="${GPU:-T4}"
HIGH_MEM="${HIGH_MEM:-0}"
REUSE_SESSION="${REUSE_SESSION:-0}"
STOP_ON_FINISH="${STOP_ON_FINISH:-0}"

if ! command -v colab >/dev/null 2>&1; then
    echo "ERROR: 'colab' CLI not found." >&2
    echo "Install it first with: pip install google-colab-cli" >&2
    exit 1
fi

if [ "$REUSE_SESSION" != "1" ]; then
    NEW_ARGS=(-s "$SESSION")
    if [ -n "$GPU" ]; then
        NEW_ARGS+=(--gpu "$GPU")
    fi
    if [ "$HIGH_MEM" = "1" ]; then
        NEW_ARGS+=(--high-mem)
    fi
    echo "==> Creating Colab session '$SESSION' (${GPU:-CPU})..."
    colab new "${NEW_ARGS[@]}"
fi

echo "==> Mounting Google Drive on session '$SESSION'..."
colab drivemount -s "$SESSION"

echo "==> Running '$SCRIPT' on Colab with DRIVE_ROOT='$DRIVE_ROOT'..."
PAYLOAD="$(
    REPO_URL="$REPO_URL" \
    BRANCH="$BRANCH" \
    REMOTE_REPO_DIR="$REMOTE_REPO_DIR" \
    DRIVE_ROOT="$DRIVE_ROOT" \
    SCRIPT="$SCRIPT" \
    MODE="${MODE:-}" \
    POISON_SETS="${POISON_SETS:-}" \
    FLAG_LEVELS="${FLAG_LEVELS:-}" \
    NUM_SEEDS="${NUM_SEEDS:-}" \
    CONFIG="${CONFIG:-}" \
    DATA_ROOT="${DATA_ROOT:-}" \
    LOG_DIR="${LOG_DIR:-}" \
    python3 -c '
import json, os
keys = [
    "REPO_URL", "BRANCH", "REMOTE_REPO_DIR", "DRIVE_ROOT", "SCRIPT",
    "MODE", "POISON_SETS", "FLAG_LEVELS", "NUM_SEEDS", "CONFIG",
    "DATA_ROOT", "LOG_DIR",
]
print(json.dumps({k: os.environ.get(k, "") for k in keys}))
'
)"

colab exec -s "$SESSION" <<EOF
import json
import os
import shlex
import subprocess
import sys

cfg = json.loads(r'''${PAYLOAD}''')
repo_url = cfg["REPO_URL"]
branch = cfg["BRANCH"]
repo_dir = cfg["REMOTE_REPO_DIR"]
drive_root = cfg["DRIVE_ROOT"]
script = cfg["SCRIPT"]

def run(cmd, cwd=None, env=None):
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=env)

# 1. Verify Drive root exists
if not os.path.isdir(drive_root):
    raise FileNotFoundError(
        f"DRIVE_ROOT not found on Colab: {drive_root}. "
        "Check that Google Drive is mounted and the path is correct."
    )

# 2. Clone or update repo
if os.path.isdir(os.path.join(repo_dir, ".git")):
    run(f"git fetch origin && git checkout {shlex.quote(branch)} && git pull origin {shlex.quote(branch)}", cwd=repo_dir)
else:
    run(f"git clone --branch {shlex.quote(branch)} {shlex.quote(repo_url)} {shlex.quote(repo_dir)}")

# 3. Install project dependencies (torch/torchvision are preinstalled on Colab)
deps = (
    "lightning>=2.0.0 torchmetrics>=0.11.0 transformers>=4.30.0 "
    "tokenizers>=0.13.0 sentencepiece>=0.1.99 hf_transfer>=0.1.9 "
    "pyarrow python-dotenv pyyaml tqdm scikit-learn plotly seaborn matplotlib"
)
run(f"{shlex.quote(sys.executable)} -m pip install -q {deps}")

# 4. Run training script with DRIVE_ROOT and forwarded env vars
env = os.environ.copy()
env["DRIVE_ROOT"] = drive_root
for key in ("MODE", "POISON_SETS", "FLAG_LEVELS", "NUM_SEEDS", "CONFIG", "DATA_ROOT", "LOG_DIR"):
    if cfg.get(key):
        env[key] = cfg[key]

run(f"bash {shlex.quote(script)}", cwd=repo_dir, env=env)
EOF

if [ "$STOP_ON_FINISH" = "1" ]; then
    echo "==> Stopping Colab session '$SESSION'..."
    colab stop "$SESSION"
else
    echo "==> Done. Session '$SESSION' is still running (stop it with: colab stop $SESSION)"
fi
