#!/bin/bash
# 自动化测试不同点数和线程数下的蒙特卡洛π估计程序

executable="./monte"

# 编译
gcc -o $executable monte.c -lpthread -lm
if [ $? -ne 0 ]; then
    echo "编译失败，请检查代码。"
    exit 1
fi

# 定义总采样点数数组（1024到65536）
n_values=(1024 4096 16384 65536)
# 定义线程数数组
thread_counts=(1 2 4 8 16)

# 定义日志文件
log_file="run_tests.log"
echo "蒙特卡洛π估计测试记录" > "$log_file"
echo "测试日期：$(date)" >> "$log_file"
echo "---------------------------------" >> "$log_file"

# 测试
for n in "${n_values[@]}"; do
    echo "测试总采样点数：${n}" | tee -a "$log_file"
    for threads in "${thread_counts[@]}"; do
        echo "  线程数：${threads}" | tee -a "$log_file"
        # 运行程序，并将输出存入变量 result
        result=$($executable $n $threads)
        echo "$result" | tee -a "$log_file"
        echo "---------------------------------" >> "$log_file"
    done
    echo "=================================" | tee -a "$log_file"
done

echo "所有测试完成，详细日志请查看 $log_file"
