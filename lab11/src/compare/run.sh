#!/bin/bash
# 运行 im2col_opt.cu 基准：对比多模式、多矩阵大小、多块大小下的运行时间
# 结果保存到 result.csv

set -e

# 可执行名字
EXE="im2col_opt_exec"
SRC="im2col_opt.cu"

# 编译
echo "Compiling $SRC -> $EXE ..."
nvcc "$SRC" -o "$EXE" -O3
echo "Compilation done."

# 输出 CSV 文件
CSV="result.csv"
echo "mode,input_size,block_size,time_ms" > "$CSV"

# 可调整的矩阵大小列表
SIZES=(256 512 1024 2048 4096)
# 可调整的块大小列表（线程块或 tile 大小等，根据代码支持选择）
BLOCKS=(8 16 32)
# 对比模式名称，需与代码中解析的字符串一致
MODES=(basic shared reged)

# 提取 Time 的函数：从程序输出中提取时间值（单位 ms）
extract_time_ms() {
    local text="$1"
    # 提取模式：匹配 'Time: X ms' 中的 X
    # 如果代码打印格式为 'Time: 12.345 ms' 或 'Execution time: 12.345 ms'，可调整正则
    local time=$(echo "$text" | grep -Po 'Time:\s*\K[\d\.]+' | head -n1)
    echo "$time"
}

# 运行可执行并记录
run_and_record() {
    local mode=$1; shift
    local size=$1; shift
    local block=$1; shift
    local out
    # 捕获 stdout/stderr
    if ! out=$(./"$EXE" "$mode" "$size" "$block" 2>&1); then
        echo "Error running ./$EXE $mode $size $block" >&2
        echo "$out" >&2
        # 记录 -1 以示错误
        echo "-1"
    else
        # 如果需要调试，可以取消下一行注释，查看完整输出
        # echo "Output for $mode $size $block:"; echo "$out"
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
    for block in "${BLOCKS[@]}"; do
        for mode in "${MODES[@]}"; do
            echo "Running mode=$mode size=$size block=$block ..."
            t=$(run_and_record "$mode" "$size" "$block")
            echo "$mode,$size,$block,$t" >> "$CSV"
            echo " -> time: ${t} ms"
        done
    done
done

echo "All runs done. Results saved in $CSV"
