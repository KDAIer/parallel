#!/bin/bash
OUTPUT="results.txt"
echo -e "方法\t矩阵规模\t线程块大小\t时间(ms)" > $OUTPUT

# 编译所有示例
nvcc -O2 row.cu          -o row
nvcc -O2 column.cu          -o column
nvcc -O2 block.cu  -o block

# 参数
sizes=(512 1024 2048)
blocks=("8 8" "16 16" "32 32")

for N in "${sizes[@]}"; do
  for blk in "${blocks[@]}"; do
    read -r bx by <<< "$blk"

    row_time=$(./row $N $N $N $bx $by | awk '{print $2}')
    column_time=$(./column $N $N $N $bx $by | awk '{print $2}')
    tile_time=$(./block $N $N $N $bx $by | awk '{print $2}')

    echo -e "row\t${N}×${N}\t${bx}×${by}\t${row_time}"     >> $OUTPUT
    echo -e "column\t${N}×${N}\t${bx}×${by}\t${column_time}"   >> $OUTPUT
    echo -e "block\t${N}×${N}\t${bx}×${by}\t${tile_time}" >> $OUTPUT
  done
done

echo "已生成结果文件：$OUTPUT"
