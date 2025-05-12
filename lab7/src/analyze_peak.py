#!/usr/bin/env python3
import subprocess
import glob
import re
import os
import matplotlib.pyplot as plt

# 可执行文件和进程数列表
binary = "./fft_parallel"  # 你的 MPI 可执行文件
processes = [1, 2, 4, 8, 16]

# 存储结果
peak_mem = []

# 正则：抓取 massif 输出中的 mem_heap_B 和 mem_heap_allocated_B
re_heap = re.compile(r"mem_heap_B=(\d+)")
re_allocated = re.compile(r"mem_heap_allocated_B=(\d+)")

for p in processes:
    massif_out = f"massif.out.{p}"
    # 删除可能的旧文件
    try:
        os.remove(massif_out)
    except FileNotFoundError:
        pass

    # 调用 mpirun + valgrind massif
    cmd = [
        "mpirun",
        "-np",
        str(p),
        "valgrind",
        "--tool=massif",
        "--massif-out-file=" + massif_out,
        binary,
    ]
    print(f"Running with {p} processes...")
    subprocess.run(cmd, check=True)

    # 解析 massif 输出
    max_heap = 0
    max_alloc = 0
    with open(massif_out, "r") as f:
        for line in f:
            m1 = re_heap.search(line)
            m2 = re_allocated.search(line)
            if m1:
                max_heap = max(max_heap, int(m1.group(1)))
            if m2:
                max_alloc = max(max_alloc, int(m2.group(1)))

    # 转换为 MB
    peak_mem.append(max_heap / (1024**2))

# 绘图
plt.figure()
plt.plot(processes, peak_mem, marker="o")
plt.xlabel("Number of Processes")
plt.ylabel("Peak Memory (MB)")
plt.title("Peak Memory Consumption vs Parallel Scale")
plt.tight_layout()
plt.savefig("peak_memory.png")

print("Plots saved as peak_memory.png")
