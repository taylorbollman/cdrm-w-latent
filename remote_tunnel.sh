#!/usr/bin/env bash
set -euo pipefail

CODE_TUNNEL="${CODE_TUNNEL:-/snap/code/current/usr/share/code/bin/code-tunnel}"
TUNNEL_NAME="${TUNNEL_NAME:-roadmap-vm2}"
VSCODE_CLI_DATA_DIR="${VSCODE_CLI_DATA_DIR:-${HOME}/.vscode/cli}"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run this script with sudo; VS Code tunnels are installed as a user systemd service." >&2
  exit 1
fi

if [[ ! -x "${CODE_TUNNEL}" ]]; then
  echo "code-tunnel not found or not executable: ${CODE_TUNNEL}" >&2
  echo "Install VS Code first, for example: sudo snap install code --classic" >&2
  exit 1
fi

if ! systemctl --user is-active --quiet dbus.service; then
  echo "[remote-tunnel] Starting user DBus session."
  systemctl --user start dbus.service
fi

tunnel_status() {
  "${CODE_TUNNEL}" --cli-data-dir "${VSCODE_CLI_DATA_DIR}" tunnel status
}

wait_for_tunnel_connected() {
  local deadline=$((SECONDS + 30))
  local status_json

  while (( SECONDS < deadline )); do
    status_json="$(tunnel_status 2>/dev/null || true)"
    if [[ "${status_json}" == *'"tunnel":"Connected"'* ]]; then
      printf '%s\n' "${status_json}"
      return 0
    fi
    sleep 1
  done

  tunnel_status || true
  return 1
}

echo "[remote-tunnel] Current tunnel status:"
tunnel_status || true

if ! "${CODE_TUNNEL}" --cli-data-dir "${VSCODE_CLI_DATA_DIR}" tunnel user show >/dev/null 2>&1; then
  echo "[remote-tunnel] Not logged into VS Code tunnel service. Complete the GitHub device login when prompted."
  "${CODE_TUNNEL}" --cli-data-dir "${VSCODE_CLI_DATA_DIR}" tunnel user login --provider github
fi

echo "[remote-tunnel] Installing/reinstalling user service for tunnel '${TUNNEL_NAME}'."
"${CODE_TUNNEL}" --cli-data-dir "${VSCODE_CLI_DATA_DIR}" tunnel service install \
  --name "${TUNNEL_NAME}" \
  --accept-server-license-terms

systemctl --user restart code-tunnel.service

echo "[remote-tunnel] Service status:"
systemctl --user status code-tunnel --no-pager

echo "[remote-tunnel] Waiting for tunnel connection:"
if status_json="$(wait_for_tunnel_connected)"; then
  echo "${status_json}"
  tunnel_name="$(printf '%s\n' "${status_json}" | sed -n 's/.*"name":"\([^"]*\)".*/\1/p')"
  if [[ -n "${tunnel_name}" ]]; then
    echo "[remote-tunnel] Open: https://vscode.dev/tunnel/${tunnel_name}"
  fi
else
  echo "[remote-tunnel] Tunnel did not report Connected within 30s. Check logs with:" >&2
  echo "  journalctl --user -u code-tunnel --no-pager -n 120" >&2
fi

if loginctl show-user "${USER}" -p Linger 2>/dev/null | grep -q '^Linger=no$'; then
  echo "[remote-tunnel] Optional persistence step: run this once if you want the tunnel to stay up after logout:"
  echo "  sudo loginctl enable-linger ${USER}"
fi
