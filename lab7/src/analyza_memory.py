#!/usr/bin/env python3
import os
import re
import matplotlib.pyplot as plt

# 要读取的进程数列表
processes = [1, 2, 4, 8, 16]

# 两个正则：allocated 和 heap 峰值
re_alloc = re.compile(r"mem_heap_allocated_B=(\d+)")
re_heap = re.compile(r"mem_heap_B=(\d+)")

# 存放每个进程数下的总体累积分配（MB）
total_alloc_mb = []

for p in processes:
    fname = f"massif.out.{p}"
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"{fname} not found")
    max_alloc = 0
    with open(fname) as f:
        for line in f:
            m = re_alloc.search(line)
            if m:
                max_alloc = max(max_alloc, int(m.group(1)))
    # 如果 allocated 一直为 0，就改用 heap 峰值
    if max_alloc == 0:
        with open(fname) as f:
            for line in f:
                m = re_heap.search(line)
                if m:
                    max_alloc = max(max_alloc, int(m.group(1)))
    # 转为 MB 并乘以进程数
    total_alloc_mb.append((max_alloc / 1024**2) * p)

# ---- 绘图 ----

plt.figure(figsize=(8, 4))
# 柱状图
plt.bar(processes, total_alloc_mb, width=0.6, alpha=0.7, label="Bar")
# 折线图
plt.plot(processes, total_alloc_mb, marker="o", linestyle="-", label="Line")
plt.xlabel("Number of Processes")
plt.ylabel("Total Allocated (or Peak) Memory (MB)")
plt.title("Total Memory vs. Parallel Scale")
plt.xticks(processes)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("total_memory_combined.png")
print("Saved plot: total_memory_combined.png")
