#!/bin/bash
# 执行 im2col_partition 基准测试，记录 row/col/block/shared 模式下的耗时到 result.csv

set -e

EXE="im2col_partition_exec"
SRC="im2col_partition.cu"

# 编译
echo "Compiling $SRC -> $EXE..."
nvcc "$SRC" -o "$EXE" -O3
echo "Compilation done."

# 输出结果文件
CSV="result.csv"
echo "mode,input_size,block_size,time_ms" > "$CSV"

# 要测试的矩阵大小和块大小
SIZES=(256 512 1024 2048 4096)
BLOCKS=(16 32 64)

# 运行并解析输出
for size in "${SIZES[@]}"; do
  for block in "${BLOCKS[@]}"; do
    echo "Running size=$size block=$block..."
    out=$("./$EXE" "$size" "$block")
    # 每行形如 "Mode: row,  Size: 256, Block: 16, Time: 12.345678 ms"
    echo "$out" | grep '^Mode:' | while read -r line; do
      mode=$(echo "$line" | grep -Po '^Mode:\s*\K[^,]+')
      sz=$(echo "$line"   | grep -Po 'Size:\s*\K[0-9]+')
      blk=$(echo "$line"  | grep -Po 'Block:\s*\K[0-9]+')
      tm=$(echo "$line"   | grep -Po 'Time:\s*\K[0-9\.]+')
      echo "${mode},${sz},${blk},${tm}" >> "$CSV"
    done
  done
done

echo "All done. Results in $CSV"
