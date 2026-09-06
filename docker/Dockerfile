ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3
FROM ${BASE_IMAGE}

ARG USER_NAME=taylorbollman
ARG USER_UID=1000
ARG USER_GID=1000
ARG MAX_JOBS=8
ARG TORCH_CUDA_ARCH_LIST="9.0;10.0"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MAX_JOBS=${MAX_JOBS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

USER root

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg; \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --batch --yes --dearmor -o /usr/share/keyrings/cloud.google.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends google-cloud-cli; \
    rm -rf /var/lib/apt/lists/*

COPY docker/requirements-docker.txt /tmp/requirements-docker.txt

RUN set -eux; \
    if grep -Eq '^[[:space:]]*[^#[:space:]]' /tmp/requirements-docker.txt; then \
      python -m pip install --no-cache-dir -r /tmp/requirements-docker.txt; \
    fi; \
    rm -f /tmp/requirements-docker.txt

RUN set -eux; \
    if ! getent group "${USER_GID}" >/dev/null; then \
      groupadd --gid "${USER_GID}" "${USER_NAME}"; \
    fi; \
    if ! getent passwd "${USER_UID}" >/dev/null; then \
      useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USER_NAME}"; \
    fi; \
    user_home="$(getent passwd "${USER_UID}" | cut -d: -f6)"; \
    mkdir -p "${user_home}" /workspace; \
    chown -R "${USER_UID}:${USER_GID}" "${user_home}" /workspace

# Install the editable package at the same path used by the project bind mount.
# The image contains a source snapshot; local edits are visible through the mount.
COPY recurrent-transformer /workspace/cdrm-w-latent/recurrent-transformer
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation \
    -e /workspace/cdrm-w-latent/recurrent-transformer

USER ${USER_UID}:${USER_GID}
WORKDIR /workspace/cdrm-w-latent
