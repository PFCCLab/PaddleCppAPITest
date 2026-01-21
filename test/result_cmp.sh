#!/bin/bash
set -e  # 出错时退出

# using guide: ./result_cmp.sh <BUILD_PATH>
BUILD_PATH=$1

PADDLE_PATH=${BUILD_PATH}/paddle/
TORCH_PATH=${BUILD_PATH}/torch/
RESULT_FILE_PATH="/tmp/paddle_cpp_api_test/"

# 记录PADDLE_PATH下所有可执行文件到列表
echo "Collecting and executing Paddle executables..."
PADDLE_EXECUTABLES=()
for test_file in ${PADDLE_PATH}/*; do
    if [[ -x "$test_file" && -f "$test_file" ]]; then
        filename=$(basename $test_file)
        ${PADDLE_PATH}${filename}
        PADDLE_EXECUTABLES+=("$filename")
        echo "Executing Paddle test: $filename"
        $test_file
    fi
done

# 记录并执行TORCH_PATH下所有可执行文件
echo "Collecting and executing Torch executables..."
TORCH_EXECUTABLES=()
for test_file in ${TORCH_PATH}/*; do
    if [[ -x "$test_file" && -f "$test_file" ]]; then
        filename=$(basename $test_file)
        ${TORCH_PATH}${filename}
        TORCH_EXECUTABLES+=("$filename")
        echo "Executing Torch test: $filename"
        $test_file
    fi
done




echo "Paddle executables (${#PADDLE_EXECUTABLES[@]}): ${PADDLE_EXECUTABLES[@]}"
echo "Torch executables (${#TORCH_EXECUTABLES[@]}): ${TORCH_EXECUTABLES[@]}"

# 比较结果文件
echo "Comparing result files..."
for ((i=0; i<${#PADDLE_EXECUTABLES[@]}; i++)); do
    paddle_file="${RESULT_FILE_PATH}/${PADDLE_EXECUTABLES[i]}.txt"
    torch_file="${RESULT_FILE_PATH}/${TORCH_EXECUTABLES[i]}.txt"

    if [[ -f "$paddle_file" && -f "$torch_file" ]]; then
        if diff -q "$paddle_file" "$torch_file" >/dev/null; then
            echo "MATCH: ${PADDLE_EXECUTABLES[i]} and ${TORCH_EXECUTABLES[i]}"
        else
            echo "DIFFER: ${PADDLE_EXECUTABLES[i]} and ${TORCH_EXECUTABLES[i]}"
            diff "$paddle_file" "$torch_file"
        fi
    else
        echo "MISSING: ${paddle_file} or ${torch_file}"
    fi
done
