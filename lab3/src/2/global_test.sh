#!/bin/bash
# 自动化测试不同数组规模和线程数下的并行数组求和

# 可执行文件名称
executable="./num_global"

# 编译
gcc -o num_global num_global.c -lpthread -lm
if [ $? -ne 0 ]; then
    echo "编译失败，请检查代码。"
    exit 1
fi

# 数组规模
# 1M=1000000，4M=4000000，16M=16000000，64M=64000000，128M=128000000
nums_sizes=(1000000 4000000 16000000 64000000 128000000)

# 线程数数组
thread_counts=(1 2 4 8 16)

# 日志文件
log_file="num_global.log"
echo "数组规模和线程数测试记录" > "$log_file"
echo "测试日期：$(date)" >> "$log_file"
echo "---------------------------------" >> "$log_file"

# 对每种数组规模和线程数进行测试
for size in "${nums_sizes[@]}"; do
    echo "测试数组规模：${size}" | tee -a "$log_file"
    for threads in "${thread_counts[@]}"; do
        echo "  线程数：${threads}" | tee -a "$log_file"
        result=$($executable $size $threads) # 输出存入变量 result
        echo "$result" | tee -a "$log_file"
        echo "---------------------------------" >> "$log_file"
    done
    echo "=================================" | tee -a "$log_file"
done

echo "所有测试完成，详细日志请查看 $log_file"
