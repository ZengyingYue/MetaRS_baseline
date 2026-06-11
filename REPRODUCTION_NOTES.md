# MetaRS 复现说明（EarthMiss_baseline）

本目录 = MetaRS 论文作者发布的 **release 代码** + `author/`（作者真实训练产物：`Best.pth`、`config.pkl`、训练 log）。
本文档记录：**严格对比 release 代码 与 作者真实训练配置（`author/config.pkl` + log）后所做的代码改动**，使本仓库上传云服务器后能忠实复现作者结果。

日期：2026-06-11。已在本地（1 GPU + `/mnt/e` 数据副本）完成静态校验，**未做完整训练/评估**（留待云端）。

---

## 0. 数据与权威来源

- 作者 config：`author/config.pkl`（用 `scripts/parse_author_config.py` 解析，实验名 `MetaRS_2x4_15k_sar`）。
- 作者 log：`author/1776068983.6830268.log`（2 GPU × batch4 = 有效 batch 8）。
- 作者权重：`author/Best.pth`（test-best 挑选的 ckpt；decoder 末层 = `(8,128,1,1)` → num_classes=8）。
- 公开数据集仅含整图：`<city>/images/SAR`、`<city>/images/RGB`、`<city>/masks`（**1024×1024 float32**），**没有** 作者用的 `_512` 目录。

---

## 1. 严格对比结论（release 配置 vs 作者 config.pkl）

| 项 | release（改前） | 作者 config.pkl | 性质 | 处理 |
|---|---|---|---|---|
| 白化损失键名 | `begin_mmr_iter=1600` / `loss.mmr` → 运行时 `mmr_loss` | `begin_isw_iter=1600` / `loss.isw` → `isw_loss` | **同一 Instance-Selective-Whitening 机制的改名版**（函数 `instance_whitening_loss`、`CovMatrix_mmr` 都在；数值/begin 值相同）。release 代码与 release 配置内部自洽 | **保留 mmr 命名**，不改（改名属无意义 churn，且不影响 Best.pth 加载——loss key 不进 state_dict） |
| `encoder.num_classes` | 1 | 8 | encoder 无 fc 层（Best.pth 证实），该字段**未被使用**，纯 cosmetic | **改为 8**（对齐作者，零风险） |
| `optimizer.weight_decay` | 0.05 | 0.05 | 一致 | 不变 |
| 训练 batch_size | base 配置 8（但 `train.sh` 覆盖为 4） | 4（×2 GPU=有效 8） | `train.sh` 已对齐作者 | base 配置也改为 4（双保险）；`train.sh` 保持 |
| missing-modality / missingce | 无（`RandomDropModality` 定义了但 `MetaRS.forward` 未调用） | 无（log 中 `missingce` 0 次） | **一致** | 不变（注意：这反而比旧 local_v1 更贴近作者——local_v1 误加了 missingce） |
| 损失项 | `sarce/rgbce/fusece`(=`loss.ce`+prefix) + `mmr` + `mse` | `sarce/rgbce/fusece` + `isw` + `mse` | 等价 | 不变 |
| **评估/协方差数据协议** | val/test/`model.params.data` 用 `SAR`/`RGB` + `masks`（整图），eval 走 1024 滑窗 | 用 `SAR_512`/`RGB_512` + `masks_512`（**预切 512 tile**），直接推理 batch16 | **真实差异**——作者所有报告数字基于 512-tile 直接推理 | 配置改为 `_512`；新增切片脚本；eval 默认直接 tile 推理 |

### 必修 bug（release 代码原样无法训练/评估）
- **`data/EarthMiss.py: get_prompt`**：`indices.size(1)` 在 numpy 数组上调用（应 `len(indices)`）、坐标索引转置错误（`indices[0][j]`→`indices[j][0]`）、内层循环变量 `i` 覆盖外层。首个训练样本即崩。作者用的是修好的版本。→ **已修**（`points` 不被模型使用，只需不报错）。
- **`module/utils.py`**：顶层 `import mmcv` 使 `ever.registry.register_all()` 崩溃（它会自动 import `module/*.py`），训练和评估都受影响。→ **改为惰性导入**。

---

## 2. 实际代码改动清单

1. `configs/baseline/MetaRS.py`：`encoder.num_classes` 1 → **8**。
2. `configs/base/EarthMiss.py`：
   - `val`/`test` 的 `sensors` → `('SAR_512','RGB_512')`，`mask_dir` → `test_mask512_dir`（=`<city>/masks_512`）。
   - 训练 `batch_size` 8 → **4**。
3. `configs/metadata/EarthMiss.py`：
   - `root_path` 支持环境变量 `EARTHMISS_ROOT` 覆盖（默认仍为作者路径 `/data/yhzhou23/data/EarthM3/`）。
   - `build_dirs(cities, mask_name)` 增加 mask 子目录参数；新增 `test_mask512_dir`。
4. `data/EarthMiss.py`：修 `get_prompt`（numpy/tensor 兼容、索引修正、变量名修正）。
5. `module/utils.py`：`import mmcv` 改为函数内惰性导入。
6. `eval.py`：新增 `--slide`。**默认（不带 `--slide`）= 作者协议**：在 `SAR_512/RGB_512` tile 上直接推理；带 `--slide` 才走 1024 滑窗（crop512/stride341）。
7. `scripts/make_512_tiles.py`（**新增**）：把整图切成 512 tile，生成 `SAR_512/RGB_512/masks_512`（默认 3 个测试城市；2×2=4 tile/图；保持 dtype；三个子目录文件名一致以通过配对 assert）。
8. `scripts/parse_author_config.py`（**新增**）：解析 `author/config.pkl`（自定义 Unpickler stub）。
9. `scripts/train.sh` / `scripts/eval.sh`：加 `EARTHMISS_ROOT`、GPU 可覆盖、指向 `author/Best.pth`、注释说明协议。

> 未改：`module/baseline/MetaRS.py`、`module/baseline/base_metars/{mmr,model}.py`、`module/loss.py`、`train.py`、`module/baseline/base*` —— 这些与作者机制一致（含 mmr.py 的 `div(HW-1)` Bessel 校正）。

---

## 3. 本地校验结果（已通过）

- ✅ 配置导入：`encoder.num_classes=8`、`num_classes=8`、`begin_mmr_iter=1600`、`loss={ce,mmr,mse,alpha}`、`wd=0.05`、train bs=4、val/test=`SAR_512/RGB_512`+`masks_512`。
- ✅ **用作者配置构建 MetaRS 并加载 `Best.pth`：0 缺失 / 0 多余 / 0 形状不匹配**（架构与作者权重完全一致）。参数量 113.918M（作者 log 113.912M，差 0.005%，非架构差异）。
- ✅ 切片脚本：2 图 → 8 tile（4/图），三模态文件名配对一致，dataloader 配对 assert 通过。
- ✅ 全 `__getitem__` 路径（含真实 `Normalize+ToTensor`）：`img (4,512,512) float32`、`mask (512,512) int8`、points、mask_down 全部正常；`get_prompt` 修复在 numpy/tensor 两路均工作。

---

## 3b. 云端服务器环境（已搭好并验证，2026-06-11）

- 机器：`ssh root@jq1.9gpu.com -p 15470`，**2× RTX 4080 SUPER (32G)**，Ubuntu 22.04，驱动 580（cu128）。
- 代码已传至 **`/data/EarthMiss_baseline`**；Python venv 在该目录 `venv/`（python 3.10）。
- 一键搭建：`bash scripts/setup_env.sh`（apt python3.10-venv/git → venv → torch 2.11+cu128 → `requirements_repro.txt` → simplecv `--no-deps`）。
- 装好的版本与本地 `metars` env 完全一致（torch 2.11.0+cu128 / ever_beta 0.5.6 / albumentations 2.0.8 / smp 0.5.0 / simplecv git@4fa6758 / tensorboardX 2.6.5 / numpy 2.2.6）。
- **环境自检通过**（`python scripts/env_check.py`，无需数据/权重）：register_all OK、配置正确、模型在 CUDA 构建 113.918M、前向 `(1,8,512,512)` softmax=1、resnet50 预训练可下载。
- 注意坑：simplecv setup 把 `albumentations` 死锁在 0.4.2、`tensorboardX` 在 1.7 —— 必须 `--no-deps` 装 simplecv（`setup_env.sh` 已处理）；pip 会打 incompatible 警告但运行正常。
- 数据集与 `Best.pth` 由用户手动上传：数据放 `/data/yhzhou23/data/EarthM3/`（或设 `EARTHMISS_ROOT`），`Best.pth` 放 `/data/EarthMiss_baseline/author/Best.pth`。
- 本机为原生 Linux 多卡，**NCCL 正常**，无需 WSL2 的 gloo patch。

## 3c. 实跑准备中发现并处理的问题（2026-06-11）

1. **缺 `imagecodecs`**：SAR/RGB 的 tif 是 LZW 压缩，`skimage/tifffile` 读图需要它。已加入 `requirements_repro.txt`（`imagecodecs==2025.3.30`）。
2. **数据集重复文件**：上传的数据里有 11 个 `*(1).tif` 下载重复件（散落在多城市的 SAR/RGB/masks），导致 `EarthM3` 的配对 assert（SAR==RGB==mask 数量）失败。已用「仅当原件存在才删」的安全逻辑清除；清除后 13 城市全部 SAR=RGB=mask 一致。**复现时若重新下载数据需再查一次。**
3. **只保存最佳 checkpoint（磁盘有限）**：改 `train.py` —— ① 把 `SaveCheckpointCallback.func` 置为 no-op，关掉框架每 20 epoch + 训练末尾的自动整存；② 在 eval hook 里追踪最佳 mIoU，仅在刷新时调 `self.checkpoint.save(filename='Best.pth')` 覆盖保存（完整 ckpt，含 optimizer + `model` 键，与 `author/Best.pth` 同格式，可被 `eval.py` 直接加载）。③ 训练期逐 tile 预测 PNG 默认关闭（`VIZ_DURING_TRAIN=1` 可开），省盘。
4. **预检（Best.pth + 512-tile 直接推理）= mIoU 0.31219**（F1 0.4246/OA 0.5668/Kappa 0.4827），命中作者 _512 协议数字（作者 log best≈31.35）。逐类 Agricultural/Playground≈0、Water 0.84，符合 MetaRS 已知行为。

## 3d. 训练运行（2026-06-11 启动）

- 数据根：`EARTHMISS_ROOT=/data/EarthMiss_baseline/data/EarthMiss/`（数据集实际在仓库 `data/EarthMiss/` 下）。
- 启动：`setsid bash scripts/train.sh > log/train_run1.log 2>&1 < /dev/null &`（脱离 SSH 独立运行）。
- 确认：2 卡 DDP、batch/GPU=4（有效 8）、AdamW lr1e-4 **wd0.05**、15000 iters（~45.3 epoch）、loss=sarce+rgbce+fusece（iter1600 后加 mmr+mse）、~0.6s/step、ETA≈2.5h。
- checkpoint：仅 `log/EarthMiss/MetaRS/Best.pth`（最佳 mIoU，覆盖保存）。eval 在 epoch 20/40 + 末尾（对齐作者 3 个评估点，峰值预期在 epoch40≈iter13200）。
- 监控：服务器端循环每 5 分钟记一行进度到后台任务输出。

## 4. 云端复现步骤

```bash
# 0) 设数据根目录
export EARTHMISS_ROOT=/your/path/to/EarthM3   # 含 13 个城市 / images / masks

# 1) 生成作者评估协议所需的 512 tile（至少 3 个测试城市；训练 iter1600 的 MMR 协方差也读它）
python scripts/make_512_tiles.py            # 默认 NewYork/Hakodate/Callao

# 2) 直接用作者权重复现评估（512-tile 直接推理，SAR 分支）
bash scripts/eval.sh                         # ckpt=author/Best.pth, 期望 SAR mIoU ≈ 31（作者 log best ≈31.35）

# 3) 从头训练（2 GPU × batch4 = 有效 8；保持 NUM_GPUS×bs==8）
bash scripts/train.sh
#    每 20 epoch 在测试城市 512-tile 上评估（作者 log：6619→28.145, 13239→31.352, 15000→28.669）
```

### 复现注意
- **有效 batch 必须 = 8**（作者 2×4）。GPU 数变了就调 `data.train.params.batch_size` 使 `NUM_GPUS×bs==8`。
- 评估数字只有在 **512-tile 直接推理** 下才能与作者 `Best.pth`/log 直接对齐；`--slide`（1024 滑窗）是另一套协议、数值不可直接比。
- `Best.pth` 是作者按测试集逐 ckpt 挑的 test-best（非最终 15000 ckpt）。最终 ckpt-15000 ≈ 28.67；峰值在 ~13000。
- 作者的 MMR/ISW 协方差源 = **测试三城市**（transductive），本配置已忠实保留（`model.params.data` 指向 `SAR_512` 测试城市）。
- WSL2 本地若要单卡跑：NCCL 在 WSL2 报 `Cuda failure 999`，需 `DIST_BACKEND=gloo`（本仓库 train.py/eval.py **未** 打 gloo patch，因目标是 Linux 云端多卡 NCCL；如需本地单卡请自行加 patch）。
```
