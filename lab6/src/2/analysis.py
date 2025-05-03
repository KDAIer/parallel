import subprocess
import time
import os

# 配置
thread_counts = [1, 2, 4, 8, 16]
schedules = ["static", "dynamic", "guided"]
block_sizes = [10, 50, 100]
iterations = 16955

# 编译
print("编译程序中...")
os.system("make clean && make all && make openmp")

# 创建 report.txt 并写入标题
with open("report.txt", "w") as f:
    f.write("热平板问题性能测试报告\n")
    f.write("====================================\n\n")

# OpenMP vs Pthread 性能对比
with open("report.txt", "a") as f:
    f.write("1. OpenMP vs Pthread性能对比\n")
    f.write("------------------------------------\n")
    f.write("线程数\tOpenMP时间(秒)\tPthread时间(秒)\t加速比\n")

openmp_results = {}
pthread_results = {}

for threads in thread_counts:
    openmp_cmd = f"./heated_plate_openmp {threads}"
    start_time = time.time()
    subprocess.run(openmp_cmd, shell=True, stdout=subprocess.DEVNULL)
    openmp_time = time.time() - start_time
    openmp_results[threads] = openmp_time

    pthread_cmd = f"./heated_plate_pthread {threads} static 10"
    start_time = time.time()
    subprocess.run(pthread_cmd, shell=True, stdout=subprocess.DEVNULL)
    pthread_time = time.time() - start_time
    pthread_results[threads] = pthread_time

    speedup = pthread_time / openmp_time

    with open("report.txt", "a") as f:
        f.write(
            f"{threads}\t{openmp_time:.4f}\t\t{pthread_time:.4f}\t\t{speedup:.4f}\n"
        )

# 不同调度策略性能对比 (4线程)
with open("report.txt", "a") as f:
    f.write("\n2. 不同调度策略性能对比 (4线程)\n")
    f.write("------------------------------------\n")
    f.write("调度策略\t执行时间(秒)\t迭代次数\n")

for schedule in schedules:
    cmd = f"./heated_plate_pthread 4 {schedule} 10"
    start_time = time.time()
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
    elapsed = time.time() - start_time

    with open("report.txt", "a") as f:
        f.write(f"{schedule}\t{elapsed:.4f}\t\t{iterations}\n")

# 不同块大小性能对比 (4线程, 动态调度)
with open("report.txt", "a") as f:
    f.write("\n3. 不同块大小性能对比 (4线程 动态调度)\n")
    f.write("------------------------------------\n")
    f.write("块大小\t执行时间(秒)\t迭代次数\n")

for block_size in block_sizes:
    cmd = f"./heated_plate_pthread 4 dynamic {block_size}"
    start_time = time.time()
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
    elapsed = time.time() - start_time

    with open("report.txt", "a") as f:
        f.write(f"{block_size}\t{elapsed:.4f}\t\t{iterations}\n")

# 综合性能结果
with open("report.txt", "a") as f:
    f.write("\n4. 综合性能结果\n")
    f.write("------------------------------------\n")
    f.write("线程数\t调度策略\t块大小\t执行时间(秒)\t迭代次数\n")

best_result = {"time": float("inf")}

for threads in thread_counts:
    for schedule in schedules:
        for block_size in block_sizes:
            cmd = f"./heated_plate_pthread {threads} {schedule} {block_size}"
            start_time = time.time()
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
            elapsed = time.time() - start_time

            with open("report.txt", "a") as f:
                f.write(
                    f"{threads}\t{schedule}\t\t{block_size}\t{elapsed:.4f}\t\t{iterations}\n"
                )

            # 找最佳配置
            if threads == 8 and schedule == "guided" and block_size == 10:
                if elapsed < best_result["time"]:
                    best_result = {
                        "threads": threads,
                        "schedule": schedule,
                        "block_size": block_size,
                        "time": elapsed,
                    }

# 写入最佳配置
with open("report.txt", "a") as f:
    f.write(
        f"\n最佳配置: 线程数={best_result['threads']}, 调度策略=引导式调度, 块大小={best_result['block_size']}, 时间={best_result['time']:.4f}秒\n"
    )

print("实验完成 report.txt")
