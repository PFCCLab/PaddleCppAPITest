#!/bin/bash
set -u

# using guide: ./result_cmp.sh <BUILD_PATH>
BUILD_PATH=$1

PADDLE_PATH=${BUILD_PATH}/paddle/
TORCH_PATH=${BUILD_PATH}/torch/
RESULT_FILE_PATH="/tmp/paddle_cpp_api_test/"

# 保存原始终端输出，并在退出时稳定打印日志路径
LOG_FILE="${RESULT_FILE_PATH}result_cmp_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${RESULT_FILE_PATH}"
exec 3>&1 4>&2
trap 'status=$?; printf "\nDone. Full output saved to: %s\n" "$LOG_FILE" | tee -a "$LOG_FILE" >&3; exit $status' EXIT
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log file: $LOG_FILE"

collect_and_run_executables() {
    local exec_path="$1"
    local prefix="$2"
    local label="$3"
    local -n out_map="$4"

    echo "Collecting and executing ${label} executables..."
    while IFS= read -r -d '' test_file; do
        local filename
        local key

        filename=$(basename "$test_file")
        if [[ "$filename" != ${prefix}_* ]]; then
            continue
        fi

        key="${filename#${prefix}_}"
        out_map["$key"]="$filename"

        echo "Executing ${label} test: $filename"
        "$test_file"
    done < <(find "$exec_path" -maxdepth 1 -type f -perm -u+x -print0 | sort -z)
}

declare -A PADDLE_EXECUTABLES
declare -A TORCH_EXECUTABLES

collect_and_run_executables "$PADDLE_PATH" "paddle" "Paddle" PADDLE_EXECUTABLES
collect_and_run_executables "$TORCH_PATH" "torch" "Torch" TORCH_EXECUTABLES

# 比较结果文件
echo "Comparing result files..."
declare -A ALL_KEYS
has_mismatch=0

for key in "${!PADDLE_EXECUTABLES[@]}"; do
    ALL_KEYS["$key"]=1
done
for key in "${!TORCH_EXECUTABLES[@]}"; do
    ALL_KEYS["$key"]=1
done

while IFS= read -r key; do
    paddle_exec="${PADDLE_EXECUTABLES[$key]}"
    torch_exec="${TORCH_EXECUTABLES[$key]}"

    if [[ -z "$paddle_exec" || -z "$torch_exec" ]]; then
        has_mismatch=1
        echo "MISSING EXECUTABLE: key=${key}, paddle=${paddle_exec:-N/A}, torch=${torch_exec:-N/A}"
        continue
    fi

    paddle_file="${RESULT_FILE_PATH}/${paddle_exec}.txt"
    torch_file="${RESULT_FILE_PATH}/${torch_exec}.txt"

    if [[ -f "$paddle_file" && -f "$torch_file" ]]; then
        if diff -q "$paddle_file" "$torch_file" >/dev/null; then
            echo "MATCH: ${paddle_exec} and ${torch_exec}"
        else
            has_mismatch=1
            echo "DIFFER: ${paddle_exec} and ${torch_exec}"
            diff "$paddle_file" "$torch_file" || true
        fi
    else
        has_mismatch=1
        echo "MISSING RESULT FILE: ${paddle_file} or ${torch_file}"
    fi
done < <(printf '%s\n' "${!ALL_KEYS[@]}" | sort)

if [[ $has_mismatch -ne 0 ]]; then
    exit 1
fi
