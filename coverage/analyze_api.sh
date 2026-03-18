#!/usr/bin/env bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# 用法:
#   1) 在任意目录执行: bash coverage/analyze_api.sh
#   2) 或直接执行: ./coverage/analyze_api.sh
#
# 说明:
#   - 脚本会自动根据自身位置定位仓库目录，不依赖当前工作目录。
#   - 输出文件默认生成在 coverage/ 目录下:
#       coverage/api_coverage_report.txt
#       coverage/api_coverage_report.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${TEST_REPO_ROOT}/.." && pwd)"

PYTHON_SCRIPT="${SCRIPT_DIR}/analyze_api_coverage.py"
HEADER_DIR="${WORKSPACE_ROOT}/Paddle/paddle/phi/api/include/compat"
INCLUDE_DIR_MAIN="${WORKSPACE_ROOT}/Paddle/paddle/phi/api/include"
INCLUDE_DIR_COMPAT="${WORKSPACE_ROOT}/Paddle/paddle/phi/api/include/compat"
TEST_DIR="${TEST_REPO_ROOT}/test"
OUTPUT_TXT="${SCRIPT_DIR}/api_coverage_report.txt"
OUTPUT_JSON="${SCRIPT_DIR}/api_coverage_report.json"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: 未找到脚本 ${PYTHON_SCRIPT}" >&2
    exit 1
fi

if [[ ! -d "${HEADER_DIR}" ]]; then
    echo "ERROR: 未找到头文件目录 ${HEADER_DIR}" >&2
    exit 1
fi

if [[ ! -d "${TEST_DIR}" ]]; then
    echo "ERROR: 未找到测试目录 ${TEST_DIR}" >&2
    exit 1
fi

python3 "${PYTHON_SCRIPT}" \
    --header-dir "${HEADER_DIR}" \
    -I "${INCLUDE_DIR_MAIN}" \
    -I "${INCLUDE_DIR_COMPAT}" \
    --test-dir "${TEST_DIR}" \
    --output "${OUTPUT_TXT}" \
    --json "${OUTPUT_JSON}"
