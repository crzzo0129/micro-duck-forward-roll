#!/usr/bin/env bash
# Ubuntu 24.04 + Python 3.12 bootstrap. Installs dependencies; never trains.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup_cloud.sh [--install-system] [--skip-gpu-check]
  --install-system  Install Ubuntu packages using root or sudo.
  --skip-gpu-check  Prepare an image before GPU allocation; skip doctor.
  --help            Show this message without making any changes.

Default: use existing system libraries, install locked Python dependencies,
then check CUDA and the task configuration. No simulation rollout or training.
USAGE
}

install_system=0
skip_gpu=0
for arg in "$@"; do
  case "$arg" in
    --install-system) install_system=1 ;;
    --skip-gpu-check) skip_gpu=1 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
done

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ "$(uname -s)" == Linux ]] || fail 'Run this script on the Linux cloud instance.'
[[ -f /etc/os-release ]] || fail 'Cannot identify the Linux distribution.'
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 24.04 ]] ||
  fail 'This system package recipe targets Ubuntu 24.04.'

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

if (( ! skip_gpu )); then
  command -v nvidia-smi >/dev/null ||
    fail 'nvidia-smi missing. Enable the GPU/driver passthrough in the cloud platform first.'
  nvidia-smi || fail 'GPU driver is unavailable. Check cloud GPU allocation and container passthrough.'
fi

if (( install_system )); then
  command -v apt-get >/dev/null || fail 'apt-get missing.'
  elevate=()
  if (( EUID != 0 )); then
    command -v sudo >/dev/null || fail 'System packages need root or sudo; ask the platform administrator.'
    elevate=(sudo)
  fi
  "${elevate[@]}" apt-get update
  "${elevate[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential pkg-config \
    python3.12-venv python3.12-dev \
    libegl1 libgl1 libglvnd0 libgles2 libglfw3 libglib2.0-0t64 tmux
fi

command -v git >/dev/null || fail 'git missing; rerun with --install-system.'
command -v curl >/dev/null || fail 'curl missing; rerun with --install-system.'
python_bin=""
for candidate in python3.12 python3 python; do
  if command -v "$candidate" >/dev/null &&
      "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
    python_bin="$(command -v "$candidate")"
    break
  fi
done
[[ -n "$python_bin" ]] || fail 'Python 3.12 missing. Use the ubuntu24.04-py3.12 image.'

# Standalone uv avoids modifying Ubuntu's externally managed system Python.
if command -v uv >/dev/null; then
  uv_bin="$(command -v uv)"
elif [[ -x "$root_dir/.tools/uv/uv" ]]; then
  uv_bin="$root_dir/.tools/uv/uv"
else
  mkdir -p "$root_dir/.cache" "$root_dir/.tools/uv"
  curl --fail --show-error --location --retry 3 \
    https://astral.sh/uv/0.12.23/install.sh -o "$root_dir/.cache/uv-install.sh"
  env UV_INSTALL_DIR="$root_dir/.tools/uv" UV_NO_MODIFY_PATH=1 \
    sh "$root_dir/.cache/uv-install.sh"
  uv_bin="$root_dir/.tools/uv/uv"
fi

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$root_dir/.cache/uv}"
export MUJOCO_GL=egl
# Record shell-quoted paths so sourcing works even with spaces in the project path.
mkdir -p "$root_dir/.cache"
{
  printf 'export PATH=%q:"$PATH"\n' "$(dirname -- "$uv_bin")"
  printf 'export UV_HTTP_TIMEOUT=%q\n' "$UV_HTTP_TIMEOUT"
  printf 'export UV_LINK_MODE=%q\n' "$UV_LINK_MODE"
  printf 'export UV_CACHE_DIR=%q\n' "$UV_CACHE_DIR"
  printf 'export MUJOCO_GL=egl\n'
} > "$root_dir/.cache/cloud.env"

"$uv_bin" --version
"$uv_bin" sync --frozen --python "$python_bin"
if (( ! skip_gpu )); then
  "$uv_bin" run --no-sync python scripts/cloud.py doctor
fi
printf '\nEnvironment prepared. No training was started.\n'
printf 'In each new terminal, run: source %q\n' "$root_dir/.cache/cloud.env"
printf 'On the GPU instance, run smoke before training:\n'
printf '  uv run python scripts/cloud.py smoke\n'
printf '  uv run python scripts/cloud.py train --num-envs 1024\n'
