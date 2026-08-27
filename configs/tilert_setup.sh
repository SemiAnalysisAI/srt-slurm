#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly TILERT_VERSION="${TILERT_VERSION:-0.1.5.post3}"
readonly NIXL_VERSION="${TILERT_NIXL_VERSION:-1.3.1}"
readonly FASTAPI_VERSION="${TILERT_FASTAPI_VERSION:-0.141.1}"
readonly UVICORN_VERSION="${TILERT_UVICORN_VERSION:-0.52.4}"

if [[ -d /opt/conda/envs/tilert/bin ]]; then
    export PATH="/opt/conda/envs/tilert/bin:${PATH}"
fi

PYTHON="$(command -v python || command -v python3)"
readonly PYTHON

installed_version() {
    "${PYTHON}" - "$1" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    print(version(sys.argv[1]))
except PackageNotFoundError:
    pass
PY
}

install_exact() {
    local package="$1"
    local version="$2"
    shift 2
    if [[ "$(installed_version "${package}")" == "${version}" ]]; then
        return
    fi
    "${PYTHON}" -m pip install --no-cache-dir "$@" "${package}==${version}"
}

case "${TILERT_ROLE:-}" in
    decode)
        install_exact tilert "${TILERT_VERSION}"
        install_exact nixl "${NIXL_VERSION}"
        # TileRT's wheel does not declare the HTTP service dependencies used
        # by its native decode server.  Install them explicitly before the
        # import preflight so the image is made runnable deterministically.
        install_exact fastapi "${FASTAPI_VERSION}"
        install_exact uvicorn "${UVICORN_VERSION}"
        "${PYTHON}" -c 'import tilert.pd_vllm.decode_server'
        ;;
    prefill)
        "${PYTHON}" -c 'import vllm'
        install_exact tilert "${TILERT_VERSION}" --no-deps
        install_exact nixl "${NIXL_VERSION}"
        "${PYTHON}" -c 'import tilert.pd_vllm.prefill_connector'
        ;;
    router)
        install_exact tilert "${TILERT_VERSION}"
        # The native router uses the same undeclared FastAPI/Uvicorn stack as
        # the decode server and must be self-contained on the frontend host.
        install_exact fastapi "${FASTAPI_VERSION}"
        install_exact uvicorn "${UVICORN_VERSION}"
        "${PYTHON}" -c 'import tilert.pd_vllm.pd_router'
        ;;
    *)
        echo "TILERT_ROLE must be prefill, decode, or router (got ${TILERT_ROLE:-unset})" >&2
        exit 1
        ;;
esac
