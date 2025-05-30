#!/bin/bash
OUTPUT="results.txt"
echo -e "方法\t矩阵规模\t线程块大小\t时间(ms)" > $OUTPUT

# 要测的矩阵规模和线程块尺寸（bx==by）
sizes=(512 1024 2048)
dims=(8 16 32)

for d in "${dims[@]}"; do
  # 编译三份可执行：RowReg_d、ColReg_d、BlockReg_d
  nvcc -DTILE_DIM=$d -O2 row.cu   -o RowReg_$d
  nvcc -DTILE_DIM=$d -O2 column.cu   -o ColReg_$d
  nvcc -DTILE_DIM=$d -O2 block.cu -o BlockReg_$d

  for N in "${sizes[@]}"; do
    # 运行并抓取时间
    t1=$(./RowReg_$d $N $N $N $d $d | awk '{print $2}')
    t2=$(./ColReg_$d $N $N $N $d $d | awk '{print $2}')
    t3=$(./BlockReg_$d $N $N $N $d $d | awk '{print $2}')

    # 写入结果
    echo -e "RowReg\t${N}×${N}\t${d}×${d}\t${t1}" >> $OUTPUT
    echo -e "ColReg\t${N}×${N}\t${d}×${d}\t${t2}" >> $OUTPUT
    echo -e "BlockReg\t${N}×${N}\t${d}×${d}\t${t3}" >> $OUTPUT
  done
done

echo "已生成共享内存优化结果文件：$OUTPUT"
