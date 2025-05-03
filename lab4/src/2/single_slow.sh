#!/bin/bash
# 自动化测试单线程版随机方程批量处理耗时

# 源文件和可执行文件
source="single_slow.c"
exe="single_slow"

# 编译
gcc -O2 -o $exe $source -lm
if [ $? -ne 0 ]; then
    echo "❌ 编译失败，请检查 $source"
    exit 1
fi

# 测试用 M 值列表
M_values=(500000 1000000 2000000)

# 日志文件
log="single_run_tests.log"
echo "单线程随机批量方程测试记录" > $log
echo "日期：$(date)" >> $log
echo "------------------------" >> $log

# 依次运行
for M in "${M_values[@]}"; do
    echo "测试方程数量: $M" | tee -a $log
    result=$(.\/$exe $M)
    echo "$result" | tee -a $log
    echo "----------------" >> $log
done

echo "测试完成，详细日志：$log"
