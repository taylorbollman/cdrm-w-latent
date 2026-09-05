#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CDRM_ROOT:-${SCRIPT_DIR}}"
INFRA_ROOT="${CDRM_INFRA_ROOT:-${PROJECT_ROOT}}"
export CDRM_ROOT="${PROJECT_ROOT}"
export CDRM_DOCKER_IMAGE="${CDRM_DOCKER_IMAGE:-cdrm-w-latent:dev}"
export CDRM_LOCALSSD_MOUNT="${CDRM_LOCALSSD_MOUNT:-/mnt/localssd}"
export CDRM_LOCAL_RUNTIME_ROOT="${CDRM_LOCAL_RUNTIME_ROOT:-${CDRM_LOCALSSD_MOUNT}/cdrm_runtime}"
LOCALSSD_MOUNT="${CDRM_LOCALSSD_MOUNT:-/mnt/localssd}"
IMAGE="${CDRM_DOCKER_IMAGE:-cdrm-w-latent:dev}"

log() {
  printf '[start-cpu] %s\n' "$*"
}

log "Starting Docker"
sudo systemctl start docker
docker info >/dev/null

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "cdrm-w-latent checkout not found at '${PROJECT_ROOT}'." >&2
  exit 1
fi

cd "${INFRA_ROOT}"

if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  log "Docker image ${IMAGE} already exists; skipping build"
else
  log "Docker image ${IMAGE} not found; building"
  bash ./scripts/docker_build.sh
fi

if findmnt -M "${LOCALSSD_MOUNT}" >/dev/null 2>&1; then
  log "${LOCALSSD_MOUNT} already mounted; checking layout"
else
  log "${LOCALSSD_MOUNT} is not mounted; setting up local SSD RAID"
fi
LOCALSSD_MOUNT_POINT="${CDRM_LOCALSSD_MOUNT}" bash ./scripts/gcp/setup_localssd_raid0.sh --yes
df -h "${LOCALSSD_MOUNT}"

log "Opening Docker shell without GPU passthrough"
export CDRM_DOCKER_GPUS=none
export CDRM_ROOT="${PROJECT_ROOT}"
exec bash ./scripts/docker_shell.sh "$@"
