# FFT 并行分析

## 目录结构

```
parallel-lab7/                # 项目根目录
├── src/                     # 源代码与脚本
│   ├── fft_serial.cpp       # 串行 FFT 实现
│   ├── fft_parallel.cpp     # 并行 FFT 实现
│   ├── packed.cpp           # packed 实现
│   ├── analyza_memory.py    # 总体内存绘图脚本（柱状+折线）
│   ├── analyze_peak.py      # 峰值内存绘图脚本
│   ├── run_packed.sh        # packed运行示例脚本
│   └── run_parallel.sh      # 并行运行示例脚本
│
├── results/                 # 分析结果目录
│   ├── figures/             # 生成的图表
│   │   ├── peak_memory.png
│   │   ├── total_memory_combined.png
│   │   ├── memory_over_time.png
│   │   └── massif_parallel_4.png
│   ├── massif_out/          # Massif 原始输出文件
│   │   ├── massif.out.1
│   │   ├── massif.out.2
│   │   ├── massif.out.4
│   │   ├── massif.out.8
│   │   └── massif.out.16
│   ├── massif_out_txt/      # ms_print 转换后的文本
│   │   ├── massif_1.txt
│   │   ├── massif_2.txt
│   │   ├── massif_4.txt
│   │   ├── massif_8.txt
│   │   └── massif_16.txt
│   └── txt/                 # 运行日志与结果文本
│       ├── fft_serial.txt
│       ├── fft_parallel.txt
│       └── packed.txt  
│
├── report/                  # 实验报告（PDF）
│   └── lab7.pdf
└──  README.md                # 本文件
```

## 1. 环境准备

1. **激活 Conda 环境**（已安装 Miniconda/Anaconda）：

   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate
   ```
2. **安装依赖**：

   ```bash
   # Python 包
   conda install -y mpi4py matplotlib numpy

   # 系统包（以 Ubuntu/Debian 为例）
   sudo apt-get update
   sudo apt-get install -y openmpi-bin libopenmpi-dev valgrind mpich
   ```
3. **验证安装**：

   ```bash
   mpic++ --version
   valgrind --version
   python -c "import matplotlib; print(matplotlib.__version__)"
   ```

---

## 2. 编译代码

进入 `src/` 目录并执行：

```bash
cd src

# 串行版本
g++ fft_serial.cpp -o fft_serial -lm

# 并行版本
mpic++ fft_parallel.cpp -o fft_parallel -lm

# packed版本
mpic++ packed.cpp -o packed -lm
```

---

## 3. 运行 FFT 程序

```bash
# 串行版本
./fft_serial

# 并行版本（指定进程数 N）
mpirun -np 4 ./fft_parallel

# packed版本
mpirun -np 4 ./packed
# 可将 `-np` 后的数字替换为 1、2、4、8、16 等不同进程数。
```

## 4. 直接脚本运行

```bash
# 并行版本
chmod +x run_parallel.sh
./run_parallel.sh

# packed版本
chmod +x run_packed.sh
./run_packed.sh
```

---

## 5. Valgrind Massif 内存分析

### 5.1 方法1

进入 Massif 输出目录：

```bash
cd ../results/massif_out
```

对多种进程数执行 Massif 采样：

```bash
for N in 1 2 4 8 16; do
  mpirun -np $N \
    valgrind \
      --tool=massif \
      --stacks=yes \
      --massif-out-file=massif.out.$N \
    ../../src/fft_parallel
  echo "生成 massif.out.$N"
done
```

如需转换为可读文本：

```bash
for N in 1 2 4 8 16; do
  ms_print massif.out.$N > massif_${N}.txt
done
```

Massif 原始输出在 `results/massif_out/`，可读文本在 `results/massif_out_txt/`。

可视化命令：

```bash
sudo apt install massif-visualizer
massif-visualizer massif.out.xxx
```

### 5.2 方法2

直接运行 `analyze_peak.py`，该代码集成了 `massif_out`的生成

```bash
cd src
python analyze_peak.py     
```

---

## 6. 解析与绘图

Python 脚本位于 `src/`：

* `analyza_memory.py`：绘制总体内存（柱状+折线）
* `analyze_peak.py`：绘制峰值内存图

```bash
cd src
python analyza_memory.py    # 输出 `total_memory_combined.png`
python analyze_peak.py      # 输出 peak_memory.png  
```

图片保存在 `results/figures/`：

* `peak_memory.png`
* `total_memory_combined.png`

---

## 7. 结果查看

* 图表：`results/figures/` 目录
* 原始 Massif 输出：`results/massif_out/` 目录
* 可读文本：`results/massif_out_txt/` 目录
* 日志文件：`results/txt/` 目录

---

## 8. 清理

删除编译文件和结果：

```bash
cd src
rm fft_serial fft_parallel
cd ../results
rm -rf figures/* massif_out/* massif_out_txt/* txt/*
```

---
