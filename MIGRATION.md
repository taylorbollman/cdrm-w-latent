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
- The workspace is now public at
  `https://github.com/taylorbollman/cdrm-w-latent`. Its recurrent-transformer
  submodule points exclusively to the personal GitHub fork for pushes.
- GPU execution and RAID creation were not tested. Full dependency rebuilds
  were not run during the initial migration; the subsequent recurrent setup
  rebuilt the image using the existing populated image as its base.

## Subsequent setup

The roadmap archive deletion freed approximately 127 GB. Recurrent-transformer
was forked to `taylorbollman/recurrent-transformer`, added as a submodule under
`vendors/`, and integrated into the Docker image as an editable installation.
The fork retains its own history and license. The three earlier vendor snapshots
remain ordinary tracked source directories.

## Recurrent installation validation

The final `cdrm-w-latent:dev` image was rebuilt successfully. In an explicitly
CPU-only container, the editable installation resolved to the mounted local
checkout, both recurrent block classes imported, `pip check` passed, and all
five upstream `tests/config_test.py` tests passed. Python remains 3.12.3 and
PyTorch retains its original NGC build. NumPy was changed from 2.1.0 to 1.26.4
to satisfy the package's `numpy<2` requirement; Rich changed from 15.0.0 to
13.9.4 to satisfy the dependency resolution. No GPU training, evaluation or
profiling was run. No source changes were pushed to the original authors.

## Stage A relocation and GPU validation — 2026-09-06

Promoted the maintained recurrent-transformer submodule from `vendors/` to
`recurrent-transformer/`, preserving its upstream starting revision
`a21b42d2bc292edb86ed1b62cee4bcab809a9d21`. Updated the restricted Docker build
context, editable installation paths, build precondition, and documentation.
Rebuilt the populated `cdrm-w-latent:dev` image without replacing NGC PyTorch/CUDA.
The mounted top-level editable import and `pip check` now pass. The launcher
sets `PYTHONNOUSERSITE=1` so persisted user-site packages cannot shadow image
pins; user-site files were preserved.

GPU `start.sh` now reuses an existing image and mounted Local SSD instead of
rebuilding on every shell launch. Both the direct launcher and home `start.sh`
wrapper reach the required project container and successfully run `nvidia-smi`.
No RAID formatting, data deletion, GPU research training, or external logging
was performed for this environment work.

Preserved the three supplied project documents unchanged under `docs/inputs/`,
with hashes and explicit provisional-protocol status. Recorded one H100 80GB,
Python 3.12.3, and CUDA 13.3 in `docs/reports/stage-a/environment.json`.
A pristine upstream tiny naïve/tiled comparison passes its surrogate checks but
shows sparse FP32 random-cotangent gradient tolerance failures; those failures
and a higher-precision diagnostic are retained, not relabeled as passes.
See `docs/reports/stage-a/environment.md` for details and limits.
