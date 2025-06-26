#!/bin/bash
# 用法: 在 src 目录下执行 ./run.sh
# 脚本功能：
# 1. 编译 CNN.cu、im2col.cu、cuDNN.cu 为可执行文件
# 2. 对 sizes=(256 512 1024 2048 4096) 和 strides=(1 2 3) 进行循环，运行各可执行文件
# 3. 提取 “Execution time: X ms” 输出，保存到 benchmark_results.csv

set -e

# 可选：如果需要指定 gcc-7 编译器，设置 COMPILER_BIN=/usr/bin/gcc-7，否则留空
# 例如：export COMPILER_BIN=/usr/bin/gcc-7
COMPILER_BIN=${COMPILER_BIN:-}

# 可执行文件名字
EXE1="cnn_exec"
EXE2="im2col_exec"
EXE3="cudnn_exec"

# CUDA 源文件名
SRC1="CNN.cu"
SRC2="im2col.cu"
SRC3="cuDNN.cu"

# 清理旧的可执行
[ -f "$EXE1" ] && rm "$EXE1"
[ -f "$EXE2" ] && rm "$EXE2"
[ -f "$EXE3" ] && rm "$EXE3"

# 编译函数：如果 COMPILER_BIN 非空，则加 --compiler-bindir
compile_cuda() {
    local src=$1
    local out=$2
    echo "Compiling $src -> $out ..."
    if [ -n "$COMPILER_BIN" ]; then
        nvcc "$src" -o "$out" -O3 --compiler-bindir="$COMPILER_BIN" ${3:-}
    else
        nvcc "$src" -o "$out" -O3 ${3:-}
    fi
}

# 编译 Task1
compile_cuda "$SRC1" "$EXE1"
# 编译 Task2
compile_cuda "$SRC2" "$EXE2"
# 编译 Task3: 需链接 cudnn
compile_cuda "$SRC3" "$EXE3" "-lcudnn"

echo "Compilation done."

# CSV 文件
CSV="benchmark_results.csv"
# 写入表头
echo "task,input_size,stride,time_ms" > "$CSV"

# 循环参数
SIZES=(256 512 1024 2048 4096)
# SIZES=(32 64 128 256 512)
STRIDES=(1 2 3)
KERNEL_SIZE=3

# 提取时间的正则函数：从输出中 grep 时间
extract_time_ms() {
    local text="$1"
    local time

    # 1) 先匹配原始格式 “Execution time: X ms”
    time=$(echo "$text" | grep -Po 'Execution time:\s*\K[0-9.]+' | head -n1)

    # 2) 再匹配前面 FP32 平均输出 “Avg Execution Time (...): X ms”
    if [ -z "$time" ]; then
        time=$(echo "$text" | grep -Po 'Avg Execution Time[^(]*\(\s*[0-9]+\s*runs\):\s*\K[0-9.]+' | head -n1)
    fi

    # 3) 再匹配 FP16 平均输出 “Avg FP16 Time (...): X ms”
    if [ -z "$time" ]; then
        time=$(echo "$text" | grep -Po 'Avg FP16 Time[^(]*\(\s*[0-9]+\s*runs\):\s*\K[0-9.]+' | head -n1)
    fi

    # 最终输出（若都没匹配到会输出空字符串）
    echo "$time"
}


# 运行可执行并提取时间
run_and_record() {
    local exe=$1
    shift
    local args=("$@")
    # 捕获 stdout/stderr
    local out
    if ! out=$("./$exe" "${args[@]}" 2>&1); then
        echo "Error running ./$exe ${args[*]}" >&2
        echo "$out" >&2
        # 记录空或-1
        echo "-1"
    else
        local t=$(extract_time_ms "$out")
        if [ -z "$t" ]; then
            echo "Warning: 无法解析时间，输出如下:" >&2
            echo "$out" >&2
            echo "-1"
        else
            echo "$t"
        fi
    fi
}

# 主循环
for size in "${SIZES[@]}"; do
    for stride in "${STRIDES[@]}"; do
        echo "Running size=$size stride=$stride ..."
        # Task1: args: <input_size> <stride>
        t1=$(run_and_record "$EXE1" "$size" "$stride")
        echo "Task1,$size,$stride,$t1" >> "$CSV"

        # Task2: args: <input_size> <kernel_size> <stride>
        t2=$(run_and_record "$EXE2" "$size" "$KERNEL_SIZE" "$stride")
        echo "Task2,$size,$stride,$t2" >> "$CSV"

        # Task3: args: <input_size> <stride>
        t3=$(run_and_record "$EXE3" "$size" "$stride")
        echo "Task3,$size,$stride,$t3" >> "$CSV"

        echo "Completed size=$size stride=$stride: Task1=${t1} ms, Task2=${t2} ms, Task3=${t3} ms"
    done
done

echo "All benchmarks done. Results saved to $CSV"
echo "You can now run: python3 run.py"
