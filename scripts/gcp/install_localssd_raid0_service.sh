#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPT="${SCRIPT_DIR}/setup_localssd_raid0.sh"
UNIT_PATH="/etc/systemd/system/cdrm-localssd-raid0.service"

if [ "${EUID}" -ne 0 ]; then
  exec sudo "$0" "$@"
fi

[ -x "${SETUP_SCRIPT}" ] || {
  echo "Setup script is not executable: ${SETUP_SCRIPT}" >&2
  exit 1
}

cat > "${UNIT_PATH}" <<EOF
[Unit]
Description=Create and mount GCP Local SSD RAID0 for cdrm-w-latent
Documentation=file://${SCRIPT_DIR}/README.md
After=systemd-udev-settle.service
Before=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStartPre=/usr/bin/udevadm settle
ExecStart=${SETUP_SCRIPT} --yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable cdrm-localssd-raid0.service

cat <<'EOF'
Installed and enabled cdrm-localssd-raid0.service.

Start it now with:
  sudo systemctl start cdrm-localssd-raid0.service

Check status with:
  systemctl status cdrm-localssd-raid0.service --no-pager
  findmnt -T /mnt/localssd
EOF
