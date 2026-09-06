#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CDRM_ROOT:-${SCRIPT_DIR}}"
INFRA_ROOT="${CDRM_INFRA_ROOT:-${PROJECT_ROOT}}"
export CDRM_ROOT="${PROJECT_ROOT}"
export CDRM_DOCKER_IMAGE="${CDRM_DOCKER_IMAGE:-cdrm-w-latent:dev}"
export CDRM_LOCALSSD_MOUNT="${CDRM_LOCALSSD_MOUNT:-/mnt/localssd}"
export CDRM_LOCAL_RUNTIME_ROOT="${CDRM_LOCAL_RUNTIME_ROOT:-${CDRM_LOCALSSD_MOUNT}/cdrm_runtime}"

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "cdrm-w-latent checkout not found at '${PROJECT_ROOT}'." >&2
  exit 1
fi

sudo systemctl start docker
docker info >/dev/null

cd "${INFRA_ROOT}"
if docker image inspect "${CDRM_DOCKER_IMAGE}" >/dev/null 2>&1; then
  echo "[start-gpu] Using existing Docker image ${CDRM_DOCKER_IMAGE}"
else
  echo "[start-gpu] Docker image missing; building"
  bash ./scripts/docker_build.sh
fi

if findmnt -M "${CDRM_LOCALSSD_MOUNT}" >/dev/null 2>&1; then
  echo "[start-gpu] Local SSD already mounted at ${CDRM_LOCALSSD_MOUNT}"
else
  LOCALSSD_MOUNT_POINT="${CDRM_LOCALSSD_MOUNT}" bash ./scripts/gcp/setup_localssd_raid0.sh --yes
fi
df -h "${CDRM_LOCALSSD_MOUNT}"

cd "${INFRA_ROOT}"
export CDRM_ROOT="${PROJECT_ROOT}"
exec bash ./scripts/docker_shell.sh "$@"
