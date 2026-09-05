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
bash ./scripts/docker_build.sh

cd "${INFRA_ROOT}"
LOCALSSD_MOUNT_POINT="${CDRM_LOCALSSD_MOUNT}" bash ./scripts/gcp/setup_localssd_raid0.sh
LOCALSSD_MOUNT_POINT="${CDRM_LOCALSSD_MOUNT}" bash ./scripts/gcp/setup_localssd_raid0.sh --yes
df -h "${CDRM_LOCALSSD_MOUNT}"

cd "${INFRA_ROOT}"
export CDRM_ROOT="${PROJECT_ROOT}"
exec bash ./scripts/docker_shell.sh "$@"
