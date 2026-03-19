# Benchmark 图表生成指南

## 概述

当运行 benchmark 时如果因缺少 `matplotlib`/`numpy` 依赖而跳过了图表生成，可以使用本指南重新生成图表。

## 方法：使用独立图表生成脚本

我们提供了一个独立的脚本 `generate_charts_from_csv.py`，可以从现有的 CSV 文件生成所有图表。

### 前提条件

安装必要的依赖：

```bash
# 使用 pip
pip install matplotlib numpy

# 或使用 uv
uv pip install matplotlib numpy
```

### 基本用法

假设你的 CSV 文件在 `./benchmark_results` 目录：

```bash
python hack/generate_charts_from_csv.py --csv-dir ./benchmark_results
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--csv-dir` | CSV 文件所在目录（必需） | - |
| `--output-dir` | 图表输出目录 | `./charts_output` |
| `--charts` | 指定生成的图表类型：`all`, `heatmap`, `lineplot`, `barplot` | `all` |
| `--generate-report` | 生成数据解读报告（Markdown） | - |
| `--format` | 输出格式：`png`, `pdf`, `svg` | `png` |
| `--console-report` | 在控制台打印报告 | `True` |

### 使用示例

#### 1. 生成所有图表

```bash
python hack/generate_charts_from_csv.py --csv-dir ./benchmark_results
```

输出：
- `heatmap_*.png` - 各 backend 和进程数的热力图
- `lineplot_throughput_vs_procs.png` - 并发性能折线图
- `barplot_backend_comparison.png` - 后端对比柱状图

#### 2. 指定输出目录

```bash
python hack/generate_charts_from_csv.py \
    --csv-dir ./benchmark_results \
    --output-dir ./my_charts
```

#### 3. 只生成特定图表

```bash
# 只生成热力图
python hack/generate_charts_from_csv.py \
    --csv-dir ./benchmark_results \
    --charts heatmap

# 生成热力图和折线图
python hack/generate_charts_from_csv.py \
    --csv-dir ./benchmark_results \
    --charts heatmap,lineplot
```

#### 4. 生成数据解读报告

```bash
python hack/generate_charts_from_csv.py \
    --csv-dir ./benchmark_results \
    --generate-report
```

这会额外生成 `data_interpretation_report.md`，包含：
- 各 backend 最佳配置分析
- 性能对比和加速比
- 可扩展性分析
- 参数敏感性分析
- 优化建议

#### 5. 生成 PDF 格式图表

```bash
python hack/generate_charts_from_csv.py \
    --csv-dir ./benchmark_results \
    --format pdf
```

### 从远端服务器获取 CSV 文件

如果 CSV 文件在远端服务器上，可以使用以下方法同步到本地：

#### 方法 1：使用 scp

```bash
# 从远端下载
scp user@remote-host:/path/to/benchmark_results/*.csv ./benchmark_results/

# 然后生成图表
python hack/generate_charts_from_csv.py --csv-dir ./benchmark_results
```

#### 方法 2：使用 rsync

```bash
rsync -avz user@remote-host:/path/to/benchmark_results/ ./benchmark_results/
python hack/generate_charts_from_csv.py --csv-dir ./benchmark_results
```

#### 方法 3：在远端直接运行

如果远端服务器有图形界面或可以保存文件：

```bash
# SSH 到远端
ssh user@remote-host

# 在远端运行
cd /path/to/project
python hack/generate_charts_from_csv.py --csv-dir ./benchmark_results

# 下载生成的图表
scp user@remote-host:/path/to/project/charts_output/* ./local_charts/
```

## 图表说明

### 1. 热力图 (heatmap_*.png)

展示不同 buffer_size 和 chunk_size 组合下的吞吐量。

- **X轴**: chunk_size (MB)
- **Y轴**: buffer_size (MB)
- **颜色**: 吞吐量 (GB/s)，越红表示越高

每个 backend 和进程数组合都会生成一张热力图。

### 2. 并发性能折线图 (lineplot_throughput_vs_procs.png)

展示各 backend 在不同进程数下的吞吐量变化。

- **X轴**: 进程数
- **Y轴**: 吞吐量 (GB/s)
- **线条**: 不同 backend 使用各自最佳配置

用于分析并行扩展性。

### 3. 后端对比柱状图 (barplot_backend_comparison.png)

对比各 backend 在最佳配置下的性能。

- **X轴**: backend 名称
- **Y轴**: 吞吐量 (GB/s)
- **标注**: 最佳配置参数

### 4. 数据解读报告 (data_interpretation_report.md)

自动生成的分析报告，包含：

- **执行摘要**: 最佳 backend 和配置
- **性能排名**: 各 backend 的对比表格
- **详细分析**: 每个 backend 的参数敏感性、可扩展性
- **优化建议**: 基于数据的配置推荐

## 故障排除

### 问题：找不到 CSV 文件

**错误信息**:
```
ERROR: Summary CSV not found: ./benchmark_results/benchmark_summary.csv
```

**解决方案**:
1. 确认 CSV 文件路径正确
2. 检查文件名是否为 `benchmark_summary.csv`

### 问题：依赖安装失败

**错误信息**:
```
ERROR: Missing required dependency: No module named 'matplotlib'
```

**解决方案**:
```bash
# 确保使用正确的 Python 环境
which python
python -m pip install matplotlib numpy

# 如果使用虚拟环境
source .venv/bin/activate  # 或 .venv\Scripts\activate (Windows)
pip install matplotlib numpy
```

### 问题：图表中文显示异常

**解决方案**:
在脚本开头添加中文字体配置：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

## 完整工作流程示例

```bash
# 1. 确保依赖已安装
pip install matplotlib numpy

# 2. 从远端同步 CSV 文件（如果需要）
scp user@server:/data/benchmark_results/*.csv ./benchmark_results/

# 3. 生成所有图表和报告
python hack/generate_charts_from_csv.py \
    --csv-dir ./benchmark_results \
    --output-dir ./charts \
    --generate-report

# 4. 查看生成的文件
ls -la ./charts/
# 输出：
#   heatmap_cpp_1procs.png
#   heatmap_cpp_4procs.png
#   ...
#   lineplot_throughput_vs_procs.png
#   barplot_backend_comparison.png
#   data_interpretation_report.md

# 5. 阅读数据解读报告
cat ./charts/data_interpretation_report.md
```

## 相关文件

- `hack/generate_charts_from_csv.py` - 独立图表生成脚本
- `hack/benchmark_report.py` - 原 benchmark 报告模块
- `hack/benchmark_runner.py` - benchmark 运行器

## 技术支持

如有问题，请检查：
1. CSV 文件格式是否正确
2. matplotlib 和 numpy 是否已正确安装
3. 输出目录是否有写入权限
