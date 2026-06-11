# MetaRS_baseline

MetaRS 论文复现仓库 - 基于 EarthMiss 数据集的多模态（光学/SAR）地物分类

## 项目概述

本仓库是 MetaRS 论文作者的 release 代码，经过与作者真实训练配置（`author/config.pkl` + 训练日志）严格对比后修改，用于云端服务器 faithful 复现作者结果。

- **实验名称**: `MetaRS_2x4_15k_sar`
- **数据集**: EarthMiss (13 个全球城市的 SAR + RGB + masks)
- **任务**: 多模态地物分类（8 类）

## 核心配置

| 配置项 | 值 |
|--------|-----|
| GPU/批次 | 2 GPU × batch 4 = **有效 batch 8** |
| 类别数 | **8** (Agriculture, Building, Water 等) |
| 训练迭代 | 15,000 (~45.3 epoch) |
| 评估协议 | **512-tile 直接推理**（作者协议） |
| 优化器 | AdamW (lr=1e-4, wd=0.05) |
| 损失函数 | sarce + rgbce + fusece + mmr(iter 1600 后) + mse |

## 环境要求

- Python 3.10
- PyTorch 2.11.0 + CUDA 12.8
- 2× GPU（推荐 RTX 4080 SUPER 32G 或更高）

```bash
# 一键环境搭建
bash scripts/setup_env.sh

# 或手动安装
pip install -r requirements_repro.txt
pip install git+https://github.com/Z-Zheng/SimpleCV.git --no-deps
pip install git+https://github.com/Z-Zheng/ever.git
```

## 数据准备

1. **下载数据集**: [Zenodo](https://zenodo.org/records/17231107) 或 [百度网盘](https://pan.baidu.com/s/1fBf4SUMssbY-gH9qPDZyrw?pwd=jfsq)

2. **设置数据路径**:
```bash
export EARTHMISS_ROOT=/your/path/to/EarthM3
```

3. **生成 512 tiles**（作者评估协议必需）:
```bash
python scripts/make_512_tiles.py
```

## 复现步骤

### 1. 评估作者模型

```bash
bash scripts/eval.sh
# 期望 SAR mIoU ≈ 31（作者 log best ≈ 31.35）
```

### 2. 从头训练

```bash
bash scripts/train.sh
# 训练时间：约 2.5 小时
# Checkpoint 保存至：log/EarthMiss/MetaRS/Best.pth
```

### 3. 监控训练进度

```bash
# 实时查看训练日志
tail -n 100 -f log/train_run1.log

# TensorBoard 可视化
tensorboard --logdir log/EarthMiss --port 6006 --bind_all
```

## 代码改动说明

与原始 release 代码相比，本仓库做了以下修改以对齐作者真实配置：

1. **`configs/baseline/MetaRS.py`**: `encoder.num_classes` 1 → 8
2. **`configs/base/EarthMiss.py`**: 
   - val/test 传感器改为 `SAR_512/RGB_512`
   - 训练 batch_size 8 → 4
3. **`configs/metadata/EarthMiss.py`**: 支持 `EARTHMISS_ROOT` 环境变量
4. **`data/EarthMiss.py`**: 修复 `get_prompt` 函数的 numpy/tensor 兼容性问题
5. **`module/utils.py`**: `import mmcv` 改为惰性导入
6. **`eval.py`**: 新增 `--slide` 选项（默认 512-tile 直接推理）
7. **新增脚本**:
   - `scripts/make_512_tiles.py`: 切片脚本
   - `scripts/parse_author_config.py`: 作者配置解析

详见 [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md)

## 复现结果预检

使用 `author/Best.pth` 在 512-tile 直接推理：

| 指标 | 值 |
|------|-----|
| mIoU | **0.31219** |
| F1 | 0.4246 |
| OA | 0.5668 |
| Kappa | 0.4827 |

与作者日志报告数字（best ≈ 31.35）吻合。

## 项目结构

```
EarthMiss_baseline/
├── author/               # 作者真实训练产物
│   ├── Best.pth         # 作者最佳 checkpoint
│   ├── config.pkl       # 作者训练配置
│   └── *.log            # 训练/评估日志
├── configs/             # 训练配置
├── data/                # 数据集加载
├── module/              # 模型定义
├── scripts/             # 训练/评估脚本
├── eval.py              # 评估入口
├── train.py             # 训练入口
├── REPRODUCTION_NOTES.md # 详细复现说明
└── README.md
```

## 注意事项

- **有效 batch 必须为 8**（作者 2×4）。GPU 数量变化时需调整 `data.train.params.batch_size`
- 评估数值只有在 **512-tile 直接推理** 下才能与作者直接对齐
- `Best.pth` 是作者按测试集挑选的 test-best checkpoint，而非最终 15000 checkpoint
- 作者的 MMR 协方差源来自测试三城市（transductive），本配置已保留

## 原项目

- **作者 GitHub**: [Yi-Heng/EarthMiss](https://github.com/Yi-Heng/EarthMiss)
- EarthMiss 项目主页: https://rsidea.whu.edu.cn/EarthMiss.html
- 原始数据集: [Zenodo](https://zenodo.org/records/17231107)

## License

本项目遵循原项目许可协议。
