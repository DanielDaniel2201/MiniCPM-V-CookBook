#!/usr/bin/env bash
# Cloud inference-only launcher for MiniCPM-o WebRTC Demo
# - Runs ONLY the C++ inference HTTP service (minicpmo_cpp_http_server.py + llama-server)
# - Does NOT start LiveKit / Backend / Frontend
# - Does NOT register to a backend (backend remains on your Windows machine)
#
# Usage:
#   bash cloud_infer_only.sh start
#   bash cloud_infer_only.sh download
#   bash cloud_infer_only.sh build
#
# Env overrides (optional):
#   PYTHON_CMD       Python interpreter (>=3.9, recommended 3.11)
#   VENV_DIR         Virtualenv directory (default: .venv-cloud)
#   LLAMACPP_ROOT    llama.cpp-omni source dir (default: ./llama.cpp-omni)
#   MODEL_DIR        GGUF model dir (default: ./models/openbmb/MiniCPM-o-4_5-gguf)
#   HF_MODEL_REPO    HuggingFace repo (default: openbmb/MiniCPM-o-4_5-gguf)
#   LLM_QUANT        Quantization filename suffix (default: Q4_K_M)
#   HF_ENDPOINT      HF mirror endpoint (default: https://hf-mirror.com)
#   GITHUB_PROXY     GitHub proxy (default: https://ghfast.top)
#   CPP_SERVER_PORT  Inference HTTP port (default: 9060; health is +1)
#   CPP_MODE         duplex|simplex (default: duplex)
#   CUDA_VISIBLE_DEVICES  GPU selection (default: 0)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# This script lives inside the WebRTC_Demo directory already.
PROJECT_DIR="$SCRIPT_DIR"
CPP_SERVER_SCRIPT="$PROJECT_DIR/cpp_server/minicpmo_cpp_http_server.py"
CPP_REF_AUDIO="$PROJECT_DIR/cpp_server/assets/default_ref_audio.wav"

PYTHON_CMD="${PYTHON_CMD:-python3}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv-cloud}"

LLAMACPP_ROOT="${LLAMACPP_ROOT:-$SCRIPT_DIR/llama.cpp-omni}"
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/tc-mb/llama.cpp-omni.git}"

MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models/openbmb/MiniCPM-o-4_5-gguf}"
HF_MODEL_REPO="${HF_MODEL_REPO:-openbmb/MiniCPM-o-4_5-gguf}"
LLM_QUANT="${LLM_QUANT:-Q4_K_M}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT

GITHUB_PROXY="${GITHUB_PROXY:-https://ghfast.top}"

CPP_MODE="${CPP_MODE:-duplex}"
CPP_SERVER_PORT="${CPP_SERVER_PORT:-9060}"
CPP_HEALTH_PORT=$((CPP_SERVER_PORT + 1))

info() { echo "[INFO] $*"; }
ok() { echo "[OK]   $*"; }
err() { echo "[ERR]  $*" >&2; }

ensure_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    err "Missing required command: $cmd"
    return 1
  }
}

download_llamacpp() {
  if [[ -d "$LLAMACPP_ROOT" ]] && [[ -f "$LLAMACPP_ROOT/CMakeLists.txt" ]]; then
    ok "llama.cpp-omni exists: $LLAMACPP_ROOT"
    return 0
  fi

  ensure_cmd git

  local repo_url="$LLAMACPP_REPO"
  if [[ -n "$GITHUB_PROXY" ]] && [[ "$repo_url" == https://github.com/* ]]; then
    repo_url="${GITHUB_PROXY}/${repo_url}"
    info "Using GitHub proxy: $repo_url"
  fi

  info "Cloning llama.cpp-omni -> $LLAMACPP_ROOT"
  git clone --depth 1 "$repo_url" "$LLAMACPP_ROOT"
  ok "llama.cpp-omni download complete"
}

build_llama_server() {
  ensure_cmd cmake
  if ! command -v make >/dev/null 2>&1 && ! command -v ninja >/dev/null 2>&1; then
    err "Need make or ninja for building"
    return 1
  fi

  if [[ -x "$LLAMACPP_ROOT/build/bin/llama-server" ]]; then
    ok "llama-server already built: $LLAMACPP_ROOT/build/bin/llama-server"
    return 0
  fi

  local cmake_args="-DCMAKE_BUILD_TYPE=Release"
  if command -v nvcc >/dev/null 2>&1 || [[ -d /usr/local/cuda ]]; then
    cmake_args="$cmake_args -DGGML_CUDA=ON"
    ok "CUDA detected -> building with GGML_CUDA=ON"
  else
    info "CUDA toolkit not detected -> building CPU-only (you can still run if your build supports it)"
  fi

  info "Building llama-server (this may take a few minutes)..."
  (
    cd "$LLAMACPP_ROOT"
    cmake -B build $cmake_args
    local ncpu
    ncpu=$(nproc 2>/dev/null || echo 4)
    cmake --build build --target llama-server -j "$ncpu"
  )

  [[ -x "$LLAMACPP_ROOT/build/bin/llama-server" ]] || {
    err "Build finished but llama-server not found at $LLAMACPP_ROOT/build/bin/llama-server"
    return 1
  }
  ok "llama-server built successfully"
}

ensure_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating venv: $VENV_DIR"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  python -m pip install -U pip
  python -m pip install -U huggingface_hub
}

download_models() {
  if [[ -d "$MODEL_DIR" ]] && [[ $(find "$MODEL_DIR" -maxdepth 1 -name "*.gguf" 2>/dev/null | wc -l) -gt 0 ]]; then
    ok "Model directory has GGUF files: $MODEL_DIR"
    return 0
  fi

  ensure_venv

  info "Downloading selected model files from $HF_MODEL_REPO (quant=$LLM_QUANT)"
  mkdir -p "$MODEL_DIR"

  python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ.get("HF_MODEL_REPO")
local_dir = os.environ.get("MODEL_DIR")
llm_quant = os.environ.get("LLM_QUANT", "Q4_K_M")
endpoint = os.environ.get("HF_ENDPOINT")

allow_patterns = [
    f"MiniCPM-o-4_5-{llm_quant}.gguf",
    "vision/*",
    "audio/*",
    "tts/*",
    "token2wav-gguf/*",
    "*.md",
    ".git*",
]

print(f"Repo: {repo_id}")
print(f"Local: {local_dir}")
print(f"Quant: {llm_quant}")
print(f"Endpoint: {endpoint}")
print(f"Allow patterns: {allow_patterns}")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=allow_patterns,
    local_dir_use_symlinks=False,
    endpoint=endpoint,
)
print("Download complete")
PY

  ok "Model download complete: $MODEL_DIR"
}

start_infer() {
  [[ -f "$CPP_SERVER_SCRIPT" ]] || {
    err "Missing inference script: $CPP_SERVER_SCRIPT"
    return 1
  }

  download_llamacpp
  build_llama_server
  download_models

  ensure_venv

  local mode_flag="--simplex"
  if [[ "$CPP_MODE" == "duplex" ]]; then
    mode_flag="--duplex"
  fi

  local gpu_devices="${CUDA_VISIBLE_DEVICES:-0}"

  info "Starting inference service on 0.0.0.0:$CPP_SERVER_PORT (health port: $CPP_HEALTH_PORT)"
  info "CPP_MODE=$CPP_MODE CUDA_VISIBLE_DEVICES=$gpu_devices"

  (
    cd "$LLAMACPP_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu_devices" \
    REF_AUDIO="$CPP_REF_AUDIO" \
    python "$CPP_SERVER_SCRIPT" \
      --llamacpp-root "$LLAMACPP_ROOT" \
      --model-dir "$MODEL_DIR" \
      --port "$CPP_SERVER_PORT" \
      --gpu-devices "$gpu_devices" \
      $mode_flag
  )
}

usage() {
  cat <<EOF
Usage:
  bash cloud_infer_only.sh {start|download|build}

Commands:
  download   Download llama.cpp-omni + models (selective)
  build      Build llama-server
  start      Download/build as needed then run inference service
EOF
}

cmd="${1:-}"
case "$cmd" in
  download)
    download_llamacpp
    download_models
    ;;
  build)
    download_llamacpp
    build_llama_server
    ;;
  start)
    start_infer
    ;;
  *)
    usage
    exit 1
    ;;
esac
