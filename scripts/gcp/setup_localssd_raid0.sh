#!/usr/bin/env bash
set -euo pipefail

MOUNT_POINT="${LOCALSSD_MOUNT_POINT:-/mnt/localssd}"
MD_DEVICE="${LOCALSSD_MD_DEVICE:-/dev/md0}"
FS_TYPE="${LOCALSSD_FS_TYPE:-ext4}"
RUNTIME_ROOT="${CDRM_LOCAL_RUNTIME_ROOT:-${MOUNT_POINT}/cdrm_runtime}"
OWNER_USER="${LOCALSSD_OWNER_USER:-${SUDO_USER:-${USER:-}}}"
YES=0
FORCE_RECREATE=0
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'USAGE'
Usage: setup_localssd_raid0.sh [--yes] [--force-recreate]

Creates a RAID0 filesystem from GCP Local SSD NVMe devices:
  /dev/disk/by-id/google-local-nvme-ssd-*

Defaults:
  LOCALSSD_MD_DEVICE=/dev/md0
  LOCALSSD_MOUNT_POINT=/mnt/localssd
  LOCALSSD_FS_TYPE=ext4
  CDRM_LOCAL_RUNTIME_ROOT=/mnt/localssd/cdrm_runtime

By default this is a dry run. Pass --yes to create/format/mount.
Pass --force-recreate only when existing RAID metadata cannot be assembled and
you intentionally want to wipe the Local SSD devices.
USAGE
}

log() {
  printf '[localssd-raid0] %s\n' "$*"
}

die() {
  printf '[localssd-raid0] ERROR: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --yes)
      YES=1
      ;;
    --force-recreate)
      FORCE_RECREATE=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

if [ "${EUID}" -ne 0 ]; then
  exec sudo env \
    LOCALSSD_MOUNT_POINT="${MOUNT_POINT}" \
    LOCALSSD_MD_DEVICE="${MD_DEVICE}" \
    LOCALSSD_FS_TYPE="${FS_TYPE}" \
    CDRM_LOCAL_RUNTIME_ROOT="${RUNTIME_ROOT}" \
    LOCALSSD_OWNER_USER="${OWNER_USER:-$(id -un)}" \
    "$0" "${ORIGINAL_ARGS[@]}"
fi

command -v mdadm >/dev/null 2>&1 || die "mdadm is not installed"
command -v findmnt >/dev/null 2>&1 || die "findmnt is not installed"
command -v wipefs >/dev/null 2>&1 || die "wipefs is not installed"

if [ -z "${OWNER_USER}" ] || ! id "${OWNER_USER}" >/dev/null 2>&1; then
  OWNER_USER="taylorbollman"
fi
OWNER_GROUP="$(id -gn "${OWNER_USER}")"

case "${FS_TYPE}" in
  ext4)
    command -v mkfs.ext4 >/dev/null 2>&1 || die "mkfs.ext4 is not installed"
    ;;
  *)
    die "Unsupported LOCALSSD_FS_TYPE='${FS_TYPE}'. This script currently supports ext4."
    ;;
esac

mapfile -t DEVICE_LINKS < <(find /dev/disk/by-id -maxdepth 1 -type l -name 'google-local-nvme-ssd-*' -print | sort)
[ "${#DEVICE_LINKS[@]}" -ge 2 ] || die "Expected at least two GCP Local SSD devices, found ${#DEVICE_LINKS[@]}"

REAL_DEVICES=()
for link in "${DEVICE_LINKS[@]}"; do
  dev="$(readlink -f "${link}")"
  [ -b "${dev}" ] || die "${link} does not resolve to a block device"
  REAL_DEVICES+=("${dev}")
done

ensure_layout() {
  mkdir -p \
    "${RUNTIME_ROOT}/local_training_memory_store" \
    "${RUNTIME_ROOT}/prefetch_staging" \
    "${RUNTIME_ROOT}/ready_windows" \
    "${RUNTIME_ROOT}/local_hf_cache" \
    "${RUNTIME_ROOT}/local_xdg_cache" \
    "${RUNTIME_ROOT}/checkpoints_local" \
    "${RUNTIME_ROOT}/profiles" \
    "${RUNTIME_ROOT}/tmp"
  chown -R "${OWNER_USER}:${OWNER_GROUP}" "${MOUNT_POINT}"
  chmod 0775 "${MOUNT_POINT}" "${RUNTIME_ROOT}"
}

mount_existing_md() {
  if [ ! -b "${MD_DEVICE}" ]; then
    return 1
  fi
  if ! blkid -o value -s TYPE "${MD_DEVICE}" >/dev/null 2>&1; then
    return 1
  fi
  mkdir -p "${MOUNT_POINT}"
  mount -o noatime,nodiratime "${MD_DEVICE}" "${MOUNT_POINT}"
  ensure_layout
  log "Mounted existing ${MD_DEVICE} at ${MOUNT_POINT}"
  df -h "${MOUNT_POINT}"
}

if findmnt -M "${MOUNT_POINT}" >/dev/null 2>&1; then
  log "${MOUNT_POINT} is already mounted"
  ensure_layout
  df -h "${MOUNT_POINT}"
  exit 0
fi

if mount_existing_md; then
  exit 0
fi

if wipefs -n "${REAL_DEVICES[@]}" 2>/dev/null | grep -q 'linux_raid_member'; then
  log "Existing RAID metadata found; trying to assemble ${MD_DEVICE}"
  if mdadm --assemble "${MD_DEVICE}" "${REAL_DEVICES[@]}"; then
    mount_existing_md
    exit 0
  fi
  if [ "${FORCE_RECREATE}" -ne 1 ]; then
    die "Existing RAID metadata could not be assembled. Re-run with --force-recreate --yes only if these Local SSD contents are disposable."
  fi
fi

for dev in "${REAL_DEVICES[@]}"; do
  if lsblk -nr -o MOUNTPOINT "${dev}" | grep -q '/'; then
    die "${dev} or one of its children is mounted; refusing to continue"
  fi
done

log "Discovered GCP Local SSD devices:"
for i in "${!DEVICE_LINKS[@]}"; do
  log "  ${DEVICE_LINKS[$i]} -> ${REAL_DEVICES[$i]}"
done
log "Planned RAID device: ${MD_DEVICE}"
log "Planned mount point: ${MOUNT_POINT}"
log "Runtime root: ${RUNTIME_ROOT}"
log "Owner: ${OWNER_USER}:${OWNER_GROUP}"

if [ "${YES}" -ne 1 ]; then
  log "Dry run only. To create, format, and mount the RAID0 volume:"
  log "  sudo $0 --yes"
  log "If stale RAID metadata exists and assembly fails:"
  log "  sudo $0 --force-recreate --yes"
  exit 0
fi

log "Creating RAID0. This will erase signatures on the Local SSD devices."
for dev in "${REAL_DEVICES[@]}"; do
  mdadm --zero-superblock --force "${dev}" >/dev/null 2>&1 || true
  wipefs -a "${dev}"
done

mdadm --create "${MD_DEVICE}" \
  --level=0 \
  --raid-devices="${#REAL_DEVICES[@]}" \
  --chunk=512K \
  --force \
  --run \
  "${REAL_DEVICES[@]}"

udevadm settle || true

case "${FS_TYPE}" in
  ext4)
    mkfs.ext4 -F -m 0 -L localssd_raid0 "${MD_DEVICE}"
    ;;
esac

mkdir -p "${MOUNT_POINT}"
mount -o noatime,nodiratime "${MD_DEVICE}" "${MOUNT_POINT}"
ensure_layout

log "RAID0 Local SSD is ready"
df -h "${MOUNT_POINT}"
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL,SERIAL "${MD_DEVICE}" "${REAL_DEVICES[@]}"
cat <<EOF

Recommended environment:
  export CDRM_LOCAL_RUNTIME_ROOT=${RUNTIME_ROOT}
  export HF_HOME=${RUNTIME_ROOT}/local_hf_cache
  export HF_HUB_CACHE=\$HF_HOME/hub
  export HF_XET_CACHE=\$HF_HOME/xet
  export HF_DATASETS_CACHE=\$HF_HOME/datasets
  export XDG_CACHE_HOME=${RUNTIME_ROOT}/local_xdg_cache
  export TRITON_CACHE_DIR=\$XDG_CACHE_HOME/triton
EOF
