#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${CDRM_DOCKER_IMAGE:-cdrm-w-latent:dev}"
BASE_IMAGE="${CDRM_BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.06-py3}"
USER_NAME="${CDRM_CONTAINER_USER:-$(id -un)}"
USER_UID="$(id -u)"
USER_GID="$(id -g)"
MAX_JOBS="${MAX_JOBS:-8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;10.0}"

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

if [ ! -f "${REPO_ROOT}/vendors/recurrent-transformer/pyproject.toml" ]; then
  echo "Initialize the recurrent-transformer submodule first: git submodule update --init --recursive" >&2
  exit 1
fi

exec "${DOCKER[@]}" build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "USER_NAME=${USER_NAME}" \
  --build-arg "USER_UID=${USER_UID}" \
  --build-arg "USER_GID=${USER_GID}" \
  --build-arg "MAX_JOBS=${MAX_JOBS}" \
  --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
  -f "${REPO_ROOT}/docker/Dockerfile" \
  -t "${IMAGE}" \
  "${REPO_ROOT}"
