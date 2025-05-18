import subprocess
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from time import time
import argparse


# 编译C++程序
def compile_program():
    print("编译程序...")
    result = subprocess.run(
        ["make", "-C", os.path.dirname(os.path.abspath(__file__))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print("编译失败:")
        print(result.stderr.decode("utf-8"))
        sys.exit(1)
    print("编译成功")


# 运行性能测试
def run_performance_test(
    adj_file, test_file, thread_counts=None, parallel_strategies=None
):
    if thread_counts is None:
        thread_counts = [1, 2, 4, 8, 16]

    if parallel_strategies is None:
        parallel_strategies = [0]  # 默认策略

    # 获取程序路径
    program_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")

    results = []

    print(f"开始性能测试，使用邻接表文件: {adj_file}")
    print(f"测试文件: {test_file}")

    for strategy in parallel_strategies:
        strategy_results = []
        print(f"\n测试并行策略 {strategy}:")

        for threads in thread_counts:
            print(f"  测试 {threads} 个线程...")
            start_time = time()

            # 运行程序
            cmd = [program_path, adj_file, test_file, str(threads), str(strategy)]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode != 0:
                print(f"  运行失败 (线程数: {threads}, 策略: {strategy}):")
                print(result.stderr.decode("utf-8"))
                continue

            # 从输出中提取计算时间
            output = result.stdout.decode("utf-8")
            for line in output.split("\n"):
                if "计算时间:" in line:
                    elapsed_time = float(line.split(":")[1].strip().split()[0])
                    strategy_results.append((threads, elapsed_time))
                    break

        results.append((strategy, strategy_results))

    return results


# 绘制结果
def plot_results(results, output_file=None):
    if not results:
        print("No results to plot")
        return

    # Strategy name mapping
    strategy_names = {
        0: "dynamic scheduling",
        1: "static scheduling",
    }

    # 运行时间、加速比和效率
    plt.figure(figsize=(12, 15))

    # 运行时间
    plt.subplot(3, 1, 1)
    for strategy, strategy_results in results:
        if not strategy_results:
            continue
        threads, times = zip(*strategy_results)
        plt.plot(
            threads,
            times,
            "o-",
            linewidth=2,
            markersize=6,
            label=f'Strategy {strategy}: {strategy_names.get(strategy, "")}',
        )
    plt.xlabel("Number of Threads")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison of Different Parallel Strategies")
    plt.grid(True)
    plt.legend()

    # 加速比
    plt.subplot(3, 1, 2)
    for strategy, strategy_results in results:
        if not strategy_results:
            continue
        threads, times = zip(*strategy_results)
        base_time = times[0]
        speedup = [base_time / t for t in times]
        plt.plot(
            threads,
            speedup,
            "o-",
            linewidth=2,
            markersize=6,
            label=f'Strategy {strategy}: {strategy_names.get(strategy, "")}',
        )
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.title("Speedup Comparison of Different Parallel Strategies")
    plt.grid(True)
    plt.legend()

    # 效率
    plt.subplot(3, 1, 3)
    for strategy, strategy_results in results:
        if not strategy_results:
            continue
        threads, times = zip(*strategy_results)
        base_time = times[0]
        speedup = [base_time / t for t in times]
        efficiency = [s / t for s, t in zip(speedup, threads)]
        plt.plot(
            threads,
            efficiency,
            "o-",
            linewidth=2,
            markersize=6,
            label=f'Strategy {strategy}: {strategy_names.get(strategy, "")}',
        )
    plt.xlabel("Number of Threads")
    plt.ylabel("Efficiency")
    plt.title("Efficiency Comparison of Different Parallel Strategies")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # 保存图像
    if output_file:
        plot_path = os.path.join(output_file, "performance_plot.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to: {plot_path}")

    plt.show()

    # 输出性能测试结果
    print("\nPerformance Test Results:")
    all_data = []

    for strategy, strategy_results in results:
        if not strategy_results:
            continue
        print(f"\nStrategy {strategy}: {strategy_names.get(strategy, '')}")
        print("Threads\tRuntime (s)\tSpeedup\t\tEfficiency")
        threads, times = zip(*strategy_results)
        base_time = times[0]
        speedup = [base_time / t for t in times]
        efficiency = [s / t for s, t in zip(speedup, threads)]

        for t, time_val, sp, ef in zip(threads, times, speedup, efficiency):
            print(f"{t}\t{time_val:.6f}\t{sp:.6f}\t{ef:.6f}")
            all_data.append(
                {
                    "Parallel Strategy": strategy,
                    "Strategy Description": strategy_names.get(strategy, ""),
                    "Threads": t,
                    "Runtime (seconds)": time_val,
                    "Speedup": sp,
                    "Efficiency": ef,
                }
            )

    if output_file and all_data:
        df = pd.DataFrame(all_data)
        csv_file = os.path.join(output_file, "performance_results.csv")
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="多源最短路径搜索性能测试")
    parser.add_argument("adj_file", help="邻接表文件路径")
    parser.add_argument("test_file", help="测试文件路径")
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="要测试的线程数列表，默认为[1, 2, 4, 8, 16]",
    )
    parser.add_argument(
        "--strategies",
        type=int,
        nargs="+",
        default=[0, 1],
        help="要测试的并行策略列表，默认为[0, 1]",
    )
    parser.add_argument("--no-compile", action="store_true", help="跳过编译步骤")

    args = parser.parse_args()

    # 创建输出目录
    base_name = args.test_file.split("/")[-1].split(".")[0]
    strategies_str = "_".join(map(str, args.strategies))
    args.output = f"../result/{base_name}_strategies_{strategies_str}_results"

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # 编译程序
    if not args.no_compile:
        compile_program()

    # 运行性能测试
    results = run_performance_test(
        args.adj_file, args.test_file, args.threads, args.strategies
    )

    # 绘制结果
    plot_results(results, args.output)


if __name__ == "__main__":
    main()
