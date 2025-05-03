import matplotlib.pyplot as plt
import pandas as pd
import chardet

# Detect file encoding & read lines
with open("result.txt", "rb") as f:
    raw = f.read()
encoding = chardet.detect(raw)["encoding"]
lines = raw.decode(encoding).splitlines()

# Parsing containers
omp_vs_pthread = []
schedule_compare = []
blocksize_compare = []
summary = []

section = None
for L in lines:
    line = L.strip()
    if not line or set(line) == set("-"):
        continue

    # Section markers (they can remain Chinese)
    if line.startswith("1. OpenMP vs Pthread性能对比"):
        section = "omp_vs_pthread"
        continue
    if line.startswith("2. 不同调度策略性能对比"):
        section = "schedule"
        continue
    if line.startswith("3. 不同块大小性能对比"):
        section = "blocksize"
        continue
    if line.startswith("4. 综合性能结果"):
        section = "summary"
        continue

    parts = line.split()
    try:
        if section == "omp_vs_pthread":
            t, o, p, s = parts
            omp_vs_pthread.append(
                {
                    "threads": int(t),
                    "openmp": float(o),
                    "pthread": float(p),
                    "speedup": float(s),
                }
            )
        elif section == "schedule":
            strat, tm, iters = parts
            schedule_compare.append(
                {"strategy_raw": strat, "time_s": float(tm), "iterations": int(iters)}
            )
        elif section == "blocksize":
            bs, tm, iters = parts
            blocksize_compare.append(
                {"block_size": int(bs), "time_s": float(tm), "iterations": int(iters)}
            )
        elif section == "summary":
            th, strat, bs, tm, iters = parts
            summary.append(
                {
                    "threads": int(th),
                    "strategy_raw": strat,
                    "block_size": int(bs),
                    "time_s": float(tm),
                    "iterations": int(iters),
                }
            )
    except:
        # skip headers or malformed lines
        continue

# Build DataFrames
df_omp = pd.DataFrame(omp_vs_pthread)
df_sch = pd.DataFrame(schedule_compare)
df_blk = pd.DataFrame(blocksize_compare)
df_sum = pd.DataFrame(summary)

# Map Chinese strategy names to English
strategy_map = {
    "静态调度": "Static",
    "动态调度": "Dynamic",
    "引导式调度": "Guided",
    # if your file uses slightly different text, add those here
}

# Apply mapping
df_sch["strategy"] = df_sch["strategy_raw"].map(strategy_map)
df_sum["strategy"] = df_sum["strategy_raw"].map(strategy_map)

plt.style.use("seaborn-v0_8-darkgrid")

# 1) OpenMP vs Pthread Execution Time
plt.figure(figsize=(8, 5))
plt.plot(df_omp["threads"], df_omp["openmp"], marker="o", label="OpenMP")
plt.plot(df_omp["threads"], df_omp["pthread"], marker="s", label="Pthread")
plt.title("OpenMP vs Pthread Execution Time")
plt.xlabel("Number of Threads")
plt.ylabel("Time (s)")
plt.legend()
plt.grid()
plt.savefig("omp_vs_pthread_time.png")
plt.close()

# 2) Scheduling Strategy Performance (4 Threads)
plt.figure(figsize=(8, 5))
plt.bar(df_sch["strategy"], df_sch["time_s"])
plt.title("Scheduling Strategy Performance (4 Threads)")
plt.xlabel("Strategy")
plt.ylabel("Time (s)")
plt.grid(axis="y")
plt.savefig("schedule_comparison.png")
plt.close()

# 3) Block Size Impact (4 Threads, Dynamic)
plt.figure(figsize=(8, 5))
plt.plot(df_blk["block_size"], df_blk["time_s"], marker="o")
plt.title("Block Size Impact (4 Threads, Dynamic)")
plt.xlabel("Block Size")
plt.ylabel("Time (s)")
plt.grid()
plt.savefig("blocksize_comparison.png")
plt.close()

# 4) Performance by Block Size for 8 Threads
t = 8
sub = df_sum[df_sum["threads"] == t]
plt.figure(figsize=(8, 5))
for strat in sub["strategy"].unique():
    part = sub[sub["strategy"] == strat]
    plt.plot(part["block_size"], part["time_s"], marker="o", label=strat)
plt.title(f"Performance by Block Size (Threads={t})")
plt.xlabel("Block Size")
plt.ylabel("Time (s)")
plt.legend()
plt.grid()
plt.savefig(f"performance_{t}_threads.png")
plt.close()

print("All charts generated.")
