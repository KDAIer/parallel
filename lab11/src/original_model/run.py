#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 benchmark_results.csv，并绘制折线图和柱状图：
- 对每个 stride: 绘制 输入大小 vs 时间 的折线图和对比柱状图。
- 对每个 输入大小: 绘制 stride vs 时间 的折线图和对比柱状图。
若 time_ms 为 NA 或无法解析，将转为 NaN，不会破坏绘图，只是图中该点会被跳过。
结果保存在 plots/ 目录下。
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_results(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # 确保列类型
    df = df.copy()
    df['input_size'] = pd.to_numeric(df['input_size'], errors='coerce').astype(int)
    df['stride'] = pd.to_numeric(df['stride'], errors='coerce').astype(int)
    # time_ms 可能是字符串或 NA，转为 float，无法解析的变 NaN
    df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')

    sizes = sorted(df['input_size'].dropna().unique())
    strides = sorted(df['stride'].dropna().unique())
    tasks = sorted(df['task'].unique())

    # 对每个 stride: 输入大小 vs 时间
    for stride in strides:
        sub = df[df['stride'] == stride]
        plt.figure()
        for task in tasks:
            task_df = sub[sub['task'] == task].sort_values('input_size')
            if task_df['time_ms'].notna().any():
                plt.plot(task_df['input_size'], task_df['time_ms'], marker='o', label=task)
        plt.xlabel('Input Size')
        plt.ylabel('Time (ms)')
        plt.title(f'Execution Time vs Input Size (Stride={stride})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'line_stride_{stride}.png'))
        plt.close()

        # 对比柱状图
        plt.figure()
        x = np.arange(len(sizes))
        width = 0.2
        for i, task in enumerate(tasks):
            task_df = sub[sub['task'] == task].set_index('input_size')
            times = []
            for size in sizes:
                if size in task_df.index:
                    times.append(task_df.at[size, 'time_ms'])
                else:
                    times.append(np.nan)
            plt.bar(x + i*width, times, width, label=task)
        plt.xlabel('Input Size')
        plt.ylabel('Time (ms)')
        plt.title(f'Execution Time Comparison (Stride={stride})')
        plt.xticks(x + width*(len(tasks)-1)/2, sizes)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'bar_stride_{stride}.png'))
        plt.close()

    # 对每个输入大小: stride vs 时间
    for size in sizes:
        sub = df[df['input_size'] == size]
        plt.figure()
        for task in tasks:
            task_df = sub[sub['task'] == task].sort_values('stride')
            if task_df['time_ms'].notna().any():
                plt.plot(task_df['stride'], task_df['time_ms'], marker='o', label=task)
        plt.xlabel('Stride')
        plt.ylabel('Time (ms)')
        plt.title(f'Execution Time vs Stride (Input Size={size})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'line_size_{size}.png'))
        plt.close()

        plt.figure()
        x = np.arange(len(strides))
        width = 0.2
        for i, task in enumerate(tasks):
            task_df = sub[sub['task'] == task].set_index('stride')
            times = []
            for stride in strides:
                if stride in task_df.index:
                    times.append(task_df.at[stride, 'time_ms'])
                else:
                    times.append(np.nan)
            plt.bar(x + i*width, times, width, label=task)
        plt.xlabel('Stride')
        plt.ylabel('Time (ms)')
        plt.title(f'Execution Time Comparison (Input Size={size})')
        plt.xticks(x + width*(len(tasks)-1)/2, strides)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'bar_size_{size}.png'))
        plt.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "benchmark_results.csv")
    if not os.path.isfile(csv_path):
        print(f"未找到 {csv_path}，请先运行 run.sh 生成 CSV。")
        return

    df = pd.read_csv(csv_path)
    output_dir = os.path.join(script_dir, "plots")
    plot_results(df, output_dir)
    print(f"Plots saved in {output_dir}")

if __name__ == "__main__":
    main()
