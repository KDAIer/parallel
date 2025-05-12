#!/usr/bin/env bash
#
# run_packed.sh — 一键编译并用多进程数执行 FFT 程序，并保存结果到 txt 文件
#

# 源码和可执行文件名
SRC="packed.cpp"
BIN="packed"
OUTFILE="packed.txt"

# 想测试的进程数列表，可根据机器修改
PROCS=(1 2 4 8 16)

# 清空或创建输出文件
echo "Packed Results" > "${OUTFILE}"
echo "Date: $(date)" >> "${OUTFILE}"
echo "" >> "${OUTFILE}"

# 编译
echo "===== 编译阶段 ====="
mpic++ -O3 -std=c++11 -fopenmp -o "${BIN}" "${SRC}" -lm
if [ $? -ne 0 ]; then
  echo "编译失败，退出！" | tee -a "${OUTFILE}"
  exit 1
fi
echo "编译完成，开始测试进程数..." >> "${OUTFILE}"

# 运行并保存
for P in "${PROCS[@]}"; do
  echo "---- 使用 ${P} 个 MPI 进程 ----" | tee -a "${OUTFILE}"
  mpirun -np "${P}" ./"${BIN}" 2>&1 | tee -a "${OUTFILE}"
  echo "" >> "${OUTFILE}"
done

echo "所有测试完成，结果保存在 ${OUTFILE}" | tee -a "${OUTFILE}"
