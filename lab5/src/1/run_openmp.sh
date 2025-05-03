#!/bin/bash
# 自动化测试 OpenMP 并行矩阵乘法 性能

# 可执行文件和源文件
exe="openmp"
src="openmp.c"

# 编译
gcc -O2 -fopenmp -o $exe $src
if [ $? -ne 0 ]; then
    echo "编译失败，请检查 $src"
    exit 1
fi

# 参数配置
matrix_sizes=(128 256 512 1024 2048)
thread_counts=(1 2 4 8 16)
schedules=(default static dynamic)

# 日志文件
log_file="openmp_tests.log"
echo "OpenMP 并行矩阵乘法测试记录" > $log_file
echo "测试日期：$(date)" >> $log_file
echo "--------------------------------------------------" >> $log_file

# 测试循环
for size in "${matrix_sizes[@]}"; do
    for sched in "${schedules[@]}"; do
        echo "矩阵规模：${size}x${size}, 调度：${sched}" | tee -a $log_file
        for threads in "${thread_counts[@]}"; do
            echo -n "  线程数：$threads ... " | tee -a $log_file
            result=$(./$exe $size $size $size $threads $sched)
            echo "$result" | tee -a $log_file
        done
        echo "----------------------------------------" >> $log_file
    done
    echo "==================================================" >> $log_file
done

echo "测试完成，详情请查看 $log_file"```
