#!/bin/bash
# 自动化测试不同矩阵规模和线程数下的并行矩阵乘法

# 可执行文件名称（编译后的程序名称）
executable="./mult"

# 编译 C 程序（确保 gcc 和 pthread 库已经安装）
gcc -o mult mult.c -lpthread -lm
if [ $? -ne 0 ]; then
    echo "编译失败，请检查代码。"
    exit 1
fi

# 定义正方形矩阵大小数组
matrix_sizes=(128 256 512 1024 2048)
# 定义线程数数组
thread_counts=(1 2 4 8 16)

# 定义日志文件
log_file="run.log"
echo "矩阵规模和线程数测试记录" > "$log_file"
echo "测试日期：$(date)" >> "$log_file"
echo "---------------------------------" >> "$log_file"

# 对每种矩阵规模和线程数进行测试
for size in "${matrix_sizes[@]}"; do
    echo "测试矩阵规模：${size}x${size}" | tee -a "$log_file"
    for threads in "${thread_counts[@]}"; do
        echo "  线程数：${threads}" | tee -a "$log_file"
        # 运行程序，并将输出存入变量 result
        result=$($executable $size $size $size $threads)
        echo "$result" | tee -a "$log_file"
        echo "---------------------------------" >> "$log_file"
    done
    echo "=================================" | tee -a "$log_file"
done

echo "所有测试完成，详细日志请查看 $log_file"
