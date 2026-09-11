#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CDRM_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "cdrm-w-latent checkout not found at '${PROJECT_ROOT}'. Set CDRM_ROOT to its host path." >&2
  exit 1
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
CONTAINER_PROJECT_ROOT="/workspace/cdrm-w-latent"
DEFAULT_IMAGE="cdrm-w-latent:dev"
IMAGE="${CDRM_DOCKER_IMAGE:-${DEFAULT_IMAGE}}"
GPU_MODE="${CDRM_DOCKER_GPUS:-all}"
CONTAINER_USER="${CDRM_CONTAINER_USER:-$(id -un)}"
CACHE_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}"
CONTAINER_HOME="${CDRM_CONTAINER_HOME:-${PROJECT_ROOT}/.docker-home}"
LOCALSSD_MOUNT="${CDRM_LOCALSSD_MOUNT:-/mnt/localssd}"
LOCAL_RUNTIME_ROOT="${CDRM_LOCAL_RUNTIME_ROOT:-${LOCALSSD_MOUNT}/cdrm_runtime}"
DOCKER_ENV_FILE=""

if [ -n "${CDRM_ENV_FILE:-}" ]; then
  ENV_FILE="${CDRM_ENV_FILE}"
elif [ -f "${PROJECT_ROOT}/.env" ]; then
  ENV_FILE="${PROJECT_ROOT}/.env"
else
  ENV_FILE="${HOME}/.env"
fi

cleanup() {
  if [ -n "${DOCKER_ENV_FILE}" ]; then
    rm -f "${DOCKER_ENV_FILE}"
  fi
}
trap cleanup EXIT

mkdir -p "${CACHE_ROOT}" "${CONTAINER_HOME}"

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

if ! "${DOCKER[@]}" image inspect "${IMAGE}" >/dev/null 2>&1; then
  if [ "${IMAGE}" = "${DEFAULT_IMAGE}" ]; then
    "${SCRIPT_DIR}/docker_build.sh"
  else
    echo "Docker image '${IMAGE}' was not found. Build it first or unset CDRM_DOCKER_IMAGE." >&2
    exit 1
  fi
fi

if [ "$#" -eq 0 ]; then
  set -- bash
fi

ENV_FILE_ARGS=()
if [ -f "${ENV_FILE}" ]; then
  DOCKER_ENV_FILE="$(mktemp)"
  awk '
    /^[[:space:]]*(#|$)/ { next }
    {
      line = $0
      sub(/^[[:space:]]+/, "", line)
      sub(/[[:space:]]+$/, "", line)
      if (line ~ /^export[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=/) {
        sub(/^export[[:space:]]+/, "", line)
      }
      if (line !~ /^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=/) {
        next
      }
      key = line
      sub(/[[:space:]]*=.*/, "", key)
      value = line
      sub(/^[^=]*=/, "", value)
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      if ((value ~ /^".*"$/) || (value ~ /^\047.*\047$/)) {
        value = substr(value, 2, length(value) - 2)
      }
      print key "=" value
    }
  ' "${ENV_FILE}" > "${DOCKER_ENV_FILE}"
  chmod 600 "${DOCKER_ENV_FILE}"
  ENV_FILE_ARGS=(--env-file "${DOCKER_ENV_FILE}")
fi

TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
  TTY_ARGS=(-it)
elif [ ! -t 0 ]; then
  TTY_ARGS=(-i)
fi

GPU_ARGS=()
case "${GPU_MODE}" in
  ""|none|off|false|0)
    ;;
  all)
    GPU_ARGS=(--gpus all)
    ;;
  *)
    GPU_ARGS=(--gpus "${GPU_MODE}")
    ;;
esac

LOCALSSD_ARGS=()
RUNTIME_ENV_ARGS=()
GCLOUD_ARGS=()
CONTAINER_XDG_CACHE_HOME=/cache
CONTAINER_HF_HOME=/cache/huggingface
CONTAINER_PIP_CACHE_DIR=/cache/pip
HOST_GCLOUD_CONFIG="${CLOUDSDK_CONFIG:-${HOME}/.config/gcloud}"
CONTAINER_GCLOUD_CONFIG="/home/${CONTAINER_USER}/.config/gcloud"

if [ -d "${HOST_GCLOUD_CONFIG}" ]; then
  GCLOUD_ARGS=(
    -v "${HOST_GCLOUD_CONFIG}:${CONTAINER_GCLOUD_CONFIG}"
    -e "CLOUDSDK_CONFIG=${CONTAINER_GCLOUD_CONFIG}"
  )
fi

if findmnt -M "${LOCALSSD_MOUNT}" >/dev/null 2>&1; then
  LOCALSSD_ARGS=(-v "${LOCALSSD_MOUNT}:${LOCALSSD_MOUNT}")
  CONTAINER_XDG_CACHE_HOME="${LOCAL_RUNTIME_ROOT}/local_xdg_cache"
  CONTAINER_HF_HOME="${LOCAL_RUNTIME_ROOT}/local_hf_cache"
  CONTAINER_PIP_CACHE_DIR="${LOCAL_RUNTIME_ROOT}/local_xdg_cache/pip"
  RUNTIME_ENV_ARGS=(
    -e "CDRM_LOCAL_RUNTIME_ROOT=${LOCAL_RUNTIME_ROOT}"
    -e "HF_HUB_CACHE=${LOCAL_RUNTIME_ROOT}/local_hf_cache/hub"
    -e "HF_XET_CACHE=${LOCAL_RUNTIME_ROOT}/local_hf_cache/xet"
    -e "HF_DATASETS_CACHE=${LOCAL_RUNTIME_ROOT}/local_hf_cache/datasets"
    -e "TRITON_CACHE_DIR=${LOCAL_RUNTIME_ROOT}/local_xdg_cache/triton"
  )
fi

set +e
"${DOCKER[@]}" run --rm "${TTY_ARGS[@]}" \
  "${GPU_ARGS[@]}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HOME="/home/${CONTAINER_USER}" \
  -e "XDG_CACHE_HOME=${CONTAINER_XDG_CACHE_HOME}" \
  -e "MPLCONFIGDIR=${CONTAINER_XDG_CACHE_HOME}/matplotlib" \
  -e "HF_HOME=${CONTAINER_HF_HOME}" \
  -e "PIP_CACHE_DIR=${CONTAINER_PIP_CACHE_DIR}" \
  -e "TORCHINDUCTOR_CACHE_DIR=${CDRM_TORCHINDUCTOR_CACHE_DIR:-${CONTAINER_XDG_CACHE_HOME}/torchinductor}" \
  "${RUNTIME_ENV_ARGS[@]}" \
  "${ENV_FILE_ARGS[@]}" \
  -e PYTHONNOUSERSITE=1 \
  "${GCLOUD_ARGS[@]}" \
  -e "CDRM_ROOT=${CONTAINER_PROJECT_ROOT}" \
  -e "PYTHONPATH=${CONTAINER_PROJECT_ROOT}:${CONTAINER_PROJECT_ROOT}/vendors/apex:${CONTAINER_PROJECT_ROOT}/vendors/flash-attention:${CONTAINER_PROJECT_ROOT}/vendors/vllm" \
  -v "${PROJECT_ROOT}:${CONTAINER_PROJECT_ROOT}" \
  -v "${CACHE_ROOT}:/cache" \
  -v "${CONTAINER_HOME}:/home/${CONTAINER_USER}" \
  "${LOCALSSD_ARGS[@]}" \
  -w "${CONTAINER_PROJECT_ROOT}" \
  "${IMAGE}" \
  "$@"
STATUS=$?
exit "${STATUS}"
