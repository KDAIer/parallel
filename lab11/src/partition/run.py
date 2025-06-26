#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 result.csv，并绘制：
1) 固定块大小：row/col/block 模式随矩阵大小的折线对比
2) 固定模式：随矩阵大小，不同块大小的折线对比
3) 固定矩阵大小：各模式随块大小的柱状图对比
保存到 plots/ 目录。
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    csv = "result.csv"
    if not os.path.isfile(csv):
        print(f"未找到 {csv}，请先运行 run.sh 生成结果。")
        return

    df = pd.read_csv(csv)
    df['input_size']  = df['input_size'].astype(int)
    df['block_size']  = df['block_size'].astype(int)
    df['time_ms']     = pd.to_numeric(df['time_ms'], errors='coerce')
    df = df[df['time_ms'].notna()]

    sizes  = sorted(df['input_size'].unique())
    blocks = sorted(df['block_size'].unique())
    modes  = sorted(df['mode'].unique())

    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)

    # 1) 固定 block_size：各模式 vs input_size
    for blk in blocks:
        sub = df[df['block_size']==blk]
        plt.figure()
        for mode in modes:
            tmp = sub[sub['mode']==mode].sort_values('input_size')
            if tmp.empty: continue
            plt.plot(tmp['input_size'], tmp['time_ms'], marker='o', label=mode)
        plt.xlabel("Input Size")
        plt.ylabel("Time (ms)")
        plt.title(f"Block={blk}: Time vs Input Size")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{outdir}/line_block_{blk}.png")
        plt.close()

    # 2) 固定 mode：各 block_size vs input_size
    for mode in modes:
        sub = df[df['mode']==mode]
        plt.figure()
        for blk in blocks:
            tmp = sub[sub['block_size']==blk].sort_values('input_size')
            if tmp.empty: continue
            plt.plot(tmp['input_size'], tmp['time_ms'], marker='o', label=f"block {blk}")
        plt.xlabel("Input Size")
        plt.ylabel("Time (ms)")
        plt.title(f"Mode={mode}: Time vs Input Size")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{outdir}/line_mode_{mode}.png")
        plt.close()

    # 3) 固定 input_size：各模式 vs block_size (柱状图)
    for size in sizes:
        sub = df[df['input_size']==size]
        x = np.arange(len(blocks))
        width = 0.8 / len(modes)
        plt.figure()
        for i, mode in enumerate(modes):
            tmp = sub[sub['mode']==mode].set_index('block_size')
            times = [tmp.at[blk,'time_ms'] if blk in tmp.index else np.nan for blk in blocks]
            plt.bar(x + i*width, times, width, label=mode)
        plt.xlabel("Block Size")
        plt.ylabel("Time (ms)")
        plt.title(f"Size={size}: Time vs Block Size")
        plt.xticks(x + width*(len(modes)-1)/2, blocks)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f"{outdir}/bar_size_{size}.png")
        plt.close()

    print(f"Plots saved in {outdir}/")

if __name__=="__main__":
    main()
