#!/bin/bash
# 自动化测试 OpenMP 矩阵乘法：4 线程、2048×2048 矩阵下 static vs dynamic 调度 chunk_size 对比

exe="chunk"
src="chunk.c"

# 编译
gcc -O2 -fopenmp -o $exe $src
if [ $? -ne 0 ]; then
    echo "✖ 编译失败，请检查 $src"
    exit 1
fi

# 参数固定为 4 线程、2048×2048 矩阵
M=2048; N=2048; K=2048; THREADS=4
CHUNKS=(1 2 4 8 16)

# 准备输出
log="chunk_compare.log"
csv="chunk_compare.csv"
echo "chunk,static_time,dynamic_time" > $csv
echo "OpenMP chunk_size 对比实验" > $log
echo "测试日期：$(date)" >> $log
echo "================================" >> $log

for chunk in "${CHUNKS[@]}"; do
    echo "## chunk_size = $chunk" | tee -a $log
    out=$("./$exe" $M $N $K $THREADS $chunk)
    echo "$out" | tee -a $log

    static_t=$(echo "$out" | awk '/Static/  {print $2}')
    dynamic_t=$(echo "$out" | awk '/Dynamic/ {print $2}')
    echo "$chunk,$static_t,$dynamic_t" >> $csv

    echo "--------------------------------" >> $log
done

echo "✔ 测试完成！"
echo "日志：$log"
echo "CSV 数据：$csv"