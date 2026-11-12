#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/LEM_data2/hyunjincho/env/lem/bin/python}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/python_onet_yearly_overlap.py" "$@"
