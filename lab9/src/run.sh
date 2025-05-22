#!/bin/bash
set -e

# 编译所有程序
nvcc hello.cu -o hello
nvcc matrix_global.cu -o matrix_global
nvcc matrix_shared.cu -o matrix_shared
nvcc matrix_shared_opt.cu -o matrix_shared_opt

echo "=== CUDA Hello World ===" > results.txt
./hello 10 2 4 >> results.txt

echo -e "\n=== Matrix Transpose Benchmark ===" >> results.txt
for N in 512 1024 2048 8192 16384; do
  for B in 8 16 32; do
    echo "-- N=$N, BLOCK=${B}x${B} --" >> results.txt
    ./matrix_global $N $B >> results.txt
    ./matrix_shared $N $B >> results.txt
    ./matrix_shared_opt $N $B >> results.txt
  done
done

echo "All results have been written to results.txt"