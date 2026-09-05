# Migration review — 2026-09-05

## Preserved

- `.env`, `.env.example`, and `.gitignore_global` from the old roadmap project.
  The home and old-project `.env` files were identical. The new `.env` has mode
  0600 and is ignored by Git; values were not printed.
- Dockerfile, pinned Python requirements, restrictive Docker build context,
  build/shell scripts, RAID0 setup script and service installer.
- Home `start.sh`, `start_cpu.sh`, and `remote_tunnel.sh` implementations now
  reside here; the home files are wrappers pointing here.
- The old roadmap `vendors` directory contained three empty submodule folders.
  All 6,377 populated vendor files were copied from jepa-gpt-rl and verified
  byte-for-byte before deletion. Git metadata was excluded from those three snapshots. Commit identifiers
  and original licenses are retained under `vendors/`.
- Container-home packages/configuration were copied into `.docker-home`, with
  the currently used container-home state overlaid. Old scratch test output
  and shell history were excluded from the resulting copy.
- Existing SSD contents were retained by renaming
  `/mnt/localssd/roadmap_jepa_runtime` to `/mnt/localssd/cdrm_runtime`.

## Configuration and cleanup

- Both old top-level Git repositories already lacked remotes. All remaining
  nested submodule remotes and configured submodule URLs were removed before
  jepa-gpt-rl was deleted. The roadmap archive retains its local Git history.
- `/home/taylorbollman/jepa-gpt-rl` was deleted after vendor verification.
- `/home/taylorbollman/roadmap-jepa-gpt` was subsequently deleted, including all
  contents, after another remote-configuration check and explicit authorization.
- `.bashrc` now uses `CDRM_ROOT` and the new project/vendor paths. Open a fresh
  terminal to pick up the new defaults; already-running shells retain their
  previously exported environment.
- `cdrm-localssd-raid0.service` is installed and enabled for future boots.
  Its executable and documentation paths point here. The old unit was disabled
  and removed; no RAID formatting was performed during this migration.
- The existing image was reused to create `cdrm-w-latent:dev` with the new image
  working directory. The portable Dockerfile supports future full rebuilds.
- The VS Code tunnel service was failing with exit 203 because Snap revision
  255 was gone. Its executable now uses `/snap/code/current/.../code-tunnel`,
  and the service was restarted. Existing tunnel configuration was kept. The
  service now runs, but it reports Disconnected and requires GitHub device
  authentication. Run `bash ~/remote_tunnel.sh` and complete the login prompt
  to restore an authenticated connection.
- The obsolete `~/.cache/jepa-gpt-rl` container-home directory was removed
  after its contents were preserved in the new `.docker-home`.
- Shared user tools, authentication, caches and VS Code state were retained.
  Historical chat/log references are not active project dependencies.

## Validation

- All migrated shell scripts and `.bashrc` pass `bash -n`.
- Mocked execution checks pass for GPU and CPU arguments, SSD and fallback
  cache paths, environment parsing/0600 permissions/temp-file cleanup,
  forwarded exit codes, automatic builds and home bootstrap wrappers.
- A real CPU-only container smoke check confirms container execution,
  `/workspace/cdrm-w-latent` working directory, project mount and vendor paths.
- `.env` and `.docker-home` are ignored; `.env.example` remains trackable.
- The workspace is now prepared for publication at
  `https://github.com/taylorbollman/cdrm-w-latent`. Its recurrent-transformer
  submodule points exclusively to the personal GitHub fork for pushes.
- GPU execution and RAID creation were not tested. Full dependency rebuilds
  were not run; installed dependencies were preserved from the existing image.

## Subsequent setup

The roadmap archive deletion freed approximately 127 GB. Recurrent-transformer
was forked to `taylorbollman/recurrent-transformer`, added as a submodule under
`vendors/`, and integrated into the Docker image as an editable installation.
The fork retains its own history and license. The three earlier vendor snapshots
remain ordinary tracked source directories.
