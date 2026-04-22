#!/usr/bin/env bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

set -e

# 获取项目根目录
ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
BUILD_PATH="${ROOT_PATH}/build"
OUTPUT_DIR="${BUILD_PATH}/html_output"

echo "=== 代码覆盖率 HTML 报告生成工具 ==="
echo "项目路径: ${ROOT_PATH}"
echo "构建路径: ${BUILD_PATH}"
echo "输出目录: ${OUTPUT_DIR}"

# 检查 lcov 是否安装
if ! command -v lcov &> /dev/null; then
    echo "错误: lcov 未安装，请先安装: sudo apt-get install lcov"
    exit 1
fi

# 检查 genhtml 是否安装
if ! command -v genhtml &> /dev/null; then
    echo "错误: genhtml 未安装，请先安装: sudo apt-get install lcov"
    exit 1
fi

# 检查是否存在 .gcda 文件
GCDA_COUNT=$(find "${BUILD_PATH}" -name "*.gcda" 2>/dev/null | wc -l)
if [ "${GCDA_COUNT}" -eq 0 ]; then
    echo "警告: 未找到 .gcda 文件，请先运行测试"
    echo "提示: 1. 使用 -DENABLE_COVERAGE=ON 重新编译"
    echo "      2. 运行 ctest 执行测试"
    exit 1
fi
echo "找到 ${GCDA_COUNT} 个 .gcda 文件"

# 步骤1: 初始化覆盖率基线（可选，用于显示未覆盖的代码）
echo ""
echo ">>> 步骤1: 收集初始覆盖率基线..."
lcov --capture --initial \
    -d "${BUILD_PATH}" \
    -o "${BUILD_PATH}/coverage_base.info" \
    --rc lcov_branch_coverage=1 \
    --ignore-errors inconsistent \
    --ignore-errors source \
    2>/dev/null || true

# 步骤2: 收集测试后的覆盖率数据
echo ""
echo ">>> 步骤2: 收集测试覆盖率数据..."
lcov --capture \
    -d "${BUILD_PATH}" \
    -o "${BUILD_PATH}/coverage_test.info" \
    --rc lcov_branch_coverage=1 \
    --ignore-errors inconsistent \
    --ignore-errors source \
    --ignore-errors mismatch \
    --ignore-errors gcov \
    --ignore-errors unused

# 步骤3: 合并基线和测试覆盖率（如果基线存在）
echo ""
echo ">>> 步骤3: 合并覆盖率数据..."
if [ -f "${BUILD_PATH}/coverage_base.info" ]; then
    lcov -a "${BUILD_PATH}/coverage_base.info" \
         -a "${BUILD_PATH}/coverage_test.info" \
         -o "${BUILD_PATH}/coverage_merged.info" \
         --rc lcov_branch_coverage=1 \
         --ignore-errors inconsistent \
         2>/dev/null || cp "${BUILD_PATH}/coverage_test.info" "${BUILD_PATH}/coverage_merged.info"
else
    cp "${BUILD_PATH}/coverage_test.info" "${BUILD_PATH}/coverage_merged.info"
fi

# 步骤4a: 生成 Paddle 覆盖率报告
echo ""
echo ">>> 步骤4a: 提取 Paddle 覆盖率数据..."
lcov --extract "${BUILD_PATH}/coverage_merged.info" \
    "*/miniconda3/*/site-packages/paddle/*" \
    "${ROOT_PATH}/*" \
    -o "${BUILD_PATH}/coverage_paddle_raw.info" \
    --rc lcov_branch_coverage=1 \
    --ignore-errors inconsistent \
    --ignore-errors source \
    --ignore-errors unused 2>/dev/null || true

# 过滤 Paddle 覆盖率（移除测试文件和不需要的目录）
echo ">>> 过滤 Paddle 覆盖率数据..."
if [ -f "${BUILD_PATH}/coverage_paddle_raw.info" ]; then
    lcov --remove "${BUILD_PATH}/coverage_paddle_raw.info" \
        '*/googletest/*' \
        '*/gtest/*' \
        '*/gmock/*' \
        '*/3rd_party/*' \
        '/usr/*' \
        '/opt/*' \
        -o "${BUILD_PATH}/coverage_paddle_filtered.info" \
        --rc lcov_branch_coverage=1 \
        --ignore-errors inconsistent \
        --ignore-errors source \
        --ignore-errors unused 2>/dev/null || true
fi

# 步骤4b: 生成 PyTorch 覆盖率报告
echo ""
echo ">>> 步骤4b: 提取 PyTorch 覆盖率数据..."
lcov --extract "${BUILD_PATH}/coverage_merged.info" \
    "*/libtorch/*" \
    "${ROOT_PATH}/*" \
    -o "${BUILD_PATH}/coverage_torch_raw.info" \
    --rc lcov_branch_coverage=1 \
    --ignore-errors inconsistent \
    --ignore-errors source \
    --ignore-errors unused 2>/dev/null || true

# 过滤 PyTorch 覆盖率
echo ">>> 过滤 PyTorch 覆盖率数据..."
if [ -f "${BUILD_PATH}/coverage_torch_raw.info" ]; then
    lcov --remove "${BUILD_PATH}/coverage_torch_raw.info" \
        '*/googletest/*' \
        '*/gtest/*' \
        '*/gmock/*' \
        '*/3rd_party/*' \
        '/usr/*' \
        '/opt/*' \
        -o "${BUILD_PATH}/coverage_torch_filtered.info" \
        --rc lcov_branch_coverage=1 \
        --ignore-errors inconsistent \
        --ignore-errors source \
        --ignore-errors unused 2>/dev/null || true
fi

# 步骤4c: 过滤综合覆盖率（去除系统和第三方库）
echo ""
echo ">>> 步骤4c: 生成综合覆盖率数据..."
lcov --remove "${BUILD_PATH}/coverage_merged.info" \
    '/usr/*' \
    '/opt/*' \
    '*/3rd_party/*' \
    '*/googletest/*' \
    '*/gtest/*' \
    '*/gmock/*' \
    '*/c++/*' \
    '*/_deps/*' \
    -o "${BUILD_PATH}/coverage_filtered.info" \
    --rc lcov_branch_coverage=1 \
    --ignore-errors inconsistent \
    --ignore-errors source \
    --ignore-errors unused

# 步骤5: 生成 HTML 报告
echo ""
echo ">>> 步骤5: 生成 HTML 可视化报告..."
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/paddle"
mkdir -p "${OUTPUT_DIR}/torch"
mkdir -p "${OUTPUT_DIR}/all"

# 生成综合报告
genhtml "${BUILD_PATH}/coverage_filtered.info" \
    -o "${OUTPUT_DIR}/all" \
    --branch-coverage \
    --legend \
    --title "PaddleCppAPITest 代码覆盖率报告（综合）" \
    --highlight \
    --demangle-cpp

# 生成 Paddle 专项报告
if [ -f "${BUILD_PATH}/coverage_paddle_filtered.info" ] && [ -s "${BUILD_PATH}/coverage_paddle_filtered.info" ]; then
    genhtml "${BUILD_PATH}/coverage_paddle_filtered.info" \
        -o "${OUTPUT_DIR}/paddle" \
        --branch-coverage \
        --legend \
        --title "PaddleCppAPITest Paddle 兼容层覆盖率报告" \
        --highlight \
        --demangle-cpp
    cp "${BUILD_PATH}/coverage_paddle_filtered.info" "${OUTPUT_DIR}/paddle/coverage.info"
    echo "✓ Paddle 覆盖率报告已生成"
else
    echo "✗ 未找到 Paddle 覆盖率数据"
fi

# 生成 PyTorch 专项报告
if [ -f "${BUILD_PATH}/coverage_torch_filtered.info" ] && [ -s "${BUILD_PATH}/coverage_torch_filtered.info" ]; then
    genhtml "${BUILD_PATH}/coverage_torch_filtered.info" \
        -o "${OUTPUT_DIR}/torch" \
        --branch-coverage \
        --legend \
        --title "PaddleCppAPITest PyTorch/LibTorch 覆盖率报告" \
        --highlight \
        --demangle-cpp
    cp "${BUILD_PATH}/coverage_torch_filtered.info" "${OUTPUT_DIR}/torch/coverage.info"
    echo "✓ PyTorch 覆盖率报告已生成"
else
    echo "✗ 未找到 PyTorch 覆盖率数据"
fi

# 复制综合的 info 文件到 html_output 目录
cp "${BUILD_PATH}/coverage_filtered.info" "${OUTPUT_DIR}/all/coverage.info"

# 步骤6: 显示覆盖率摘要
echo ""
echo ">>> 步骤6: 覆盖率摘要"
echo "=============================================="
echo ""
echo "【综合覆盖率】"
lcov --list "${OUTPUT_DIR}/all/coverage.info" --rc lcov_branch_coverage=1 | tail -5
echo ""
if [ -f "${OUTPUT_DIR}/paddle/coverage.info" ]; then
    echo "【Paddle 兼容层覆盖率】"
    lcov --list "${OUTPUT_DIR}/paddle/coverage.info" --rc lcov_branch_coverage=1 | tail -5
    echo ""
fi
if [ -f "${OUTPUT_DIR}/torch/coverage.info" ]; then
    echo "【PyTorch 覆盖率】"
    lcov --list "${OUTPUT_DIR}/torch/coverage.info" --rc lcov_branch_coverage=1 | tail -5
    echo ""
fi
echo "=============================================="

echo ""
echo "=== 完成！==="
echo "HTML 报告已生成到: ${OUTPUT_DIR}"
echo ""
echo "查看报告:"
echo "  - 综合报告: ${OUTPUT_DIR}/all/index.html"
if [ -f "${OUTPUT_DIR}/paddle/index.html" ]; then
    echo "  - Paddle 报告: ${OUTPUT_DIR}/paddle/index.html"
fi
if [ -f "${OUTPUT_DIR}/torch/index.html" ]; then
    echo "  - PyTorch 报告: ${OUTPUT_DIR}/torch/index.html"
fi
echo ""
echo "或运行以下命令查看:"
echo "  xdg-open ${OUTPUT_DIR}/all/index.html"
