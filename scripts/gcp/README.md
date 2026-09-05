# GCP Local SSD bootstrap

`setup_localssd_raid0.sh` preserves the existing RAID assembly, formatting,
mounting and runtime-directory setup behavior. It discovers Google Local SSD
NVMe devices and uses `/dev/md0`, `/mnt/localssd`, and
`/mnt/localssd/cdrm_runtime` by default.

Run `bash setup_localssd_raid0.sh --help` for options. `--yes` authorizes initial
creation/formatting. Existing arrays are reused. The inherited default dry-run
mode can still assemble/mount an existing array and create its directories;
it is not a strictly read-only inspection command.

`sudo bash install_localssd_raid0_service.sh` installs and enables
`cdrm-localssd-raid0.service` for future boots. Installation does not start it.

Environment overrides: `LOCALSSD_MOUNT_POINT`, `LOCALSSD_MD_DEVICE`,
`LOCALSSD_FS_TYPE`, `LOCALSSD_OWNER_USER`, `CDRM_LOCAL_RUNTIME_ROOT`.
