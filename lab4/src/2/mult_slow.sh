#!/bin/bash
# 自动化测试随机系数流水线批量处理

source="mult_slow.c"
exe="mult_slow"

# 编译
gcc -O2 -pthread -o $exe $source -lm
if [ $? -ne 0 ]; then
    echo "编译失败，请检查 $source"
    exit 1
fi

# 测试配置
M_values=(500000 1000000 2000000)
thread_counts=(3 6 9 12)

log="mult_slow_tests.log"
echo "随机系数流水线测试" > $log
echo "日期：$(date)" >> $log
echo "------------------------" >> $log

for M in "${M_values[@]}"; do
  echo "方程数量: $M" | tee -a $log
  for t in "${thread_counts[@]}"; do
    echo "  线程数: $t" | tee -a $log
    result=$(.\/$exe $M $t)
    echo "$result" | tee -a $log
    echo "----------------" >> $log
  done
  echo "========================" | tee -a $log
done

echo "测试完成，详细日志：$log"
