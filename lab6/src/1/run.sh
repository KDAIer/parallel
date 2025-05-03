#!/bin/bash
set -e

# 清理并重建
make clean
make

# 固定矩阵尺寸
m=1024; n=1024; k=1024

# 参数列表
threads_list=(1 2 4 8 16)
chunks=(1 16 64 256 512)
schedules=(0 1 2)  # 0=static,1=dynamic,2=guided

# 输出头
echo "schedule threads chunk m n k time_s speedup" > bench_results.txt

for sched in "${schedules[@]}"; do
  for th in "${threads_list[@]}"; do
    if [ "$sched" -eq 0 ]; then
      chunk=0
      ./main $sched $th $chunk $m $n $k >> bench_results.txt
    else
      for chunk in "${chunks[@]}"; do
        ./main $sched $th $chunk $m $n $k >> bench_results.txt
      done
    fi
  done
done

echo "Done. see bench_results.txt"
