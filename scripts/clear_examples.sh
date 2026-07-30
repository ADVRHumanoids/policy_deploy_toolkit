#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXAMPLES_DIR="${PREFIX_DIR}/share/policy_deploy_toolkit/examples"

if [[ ! -d "${EXAMPLES_DIR}" ]]; then
    echo "Examples directory not found: ${EXAMPLES_DIR}"
    exit 0
fi

rm -rf "${EXAMPLES_DIR}"/*
rm -rf "${EXAMPLES_DIR}"/.[!.]* "${EXAMPLES_DIR}"/..?* 2>/dev/null || true

echo "Cleared: ${EXAMPLES_DIR}"
