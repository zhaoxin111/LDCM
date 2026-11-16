# Odin1 LDCM 离线推理

所有下述命令从 `/workspace/LDCM` 运行；新输出保存于该项目的 `outputs`。测试脚本不再导入 dToF-DMDC 的 Python 代码。

使用 `/root/miniconda3/bin/python`（base Python 3.14），不降级 Python、PyTorch，不改变 DMDC 的入口。默认 CPU、四线程、逐帧独立推理，无运动补偿和多帧融合。

源码来自 `/workspace/LDCM`，提交 `72de2ed17706b56c995a2d81c9ef6b830ee941b6`。在该目录应用本仓库的 `patches/ldcm-cpu-fp32.patch`：CPU 向 MoGe 传 FP32 并设置 `use_fp16=False`，CUDA 保留 FP16。CPU 使用上游原有 PyTorch 泊松求解器。MoGe 内部仍可能提示 CPU float32 autocast 被禁用，这是 PyTorch 自动保持 FP32 的提示。

官方指定的 utils3d 提交与 DMDC 当前依赖不同，隔离安装在本项目中：

```bash
/root/miniconda3/bin/python -m pip install --no-deps --target weights/ldcm_runtime \
  'git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183'
```

两个公开权重保存在本地，固定版本为：

| 来源 | revision | 本地路径 |
| --- | --- | --- |
| [pkqbajng/LDCM](https://huggingface.co/pkqbajng/LDCM) | `ffd7761e902f056508502167a485c9138e8378b7` | `weights/ldcm/ldcm.pt` |
| [Ruicheng/moge-2-vits-normal](https://huggingface.co/Ruicheng/moge-2-vits-normal) | `26b477f41595707c5db6770294c0d1721e8ed4ed` | `weights/moge-2-vits-normal/model.pt` |

下载方式（在 `/workspace/LDCM` 执行）：

```python
from huggingface_hub import hf_hub_download
hf_hub_download('pkqbajng/LDCM', 'ldcm.pt',
    revision='ffd7761e902f056508502167a485c9138e8378b7', local_dir='weights/ldcm')
hf_hub_download('Ruicheng/moge-2-vits-normal', 'model.pt',
    revision='26b477f41595707c5db6770294c0d1721e8ed4ed', local_dir='weights/moge-2-vits-normal')
```

先运行官方 sample_1，再运行 Odin1：

```bash
/root/miniconda3/bin/python scripts/infer_ldcm_sample.py
/root/miniconda3/bin/python scripts/infer_odin1_ldcm.py
```

输出分别为 `outputs/ldcm_sample_1` 和 `outputs/ldcm_03`。已有目录会拒绝覆盖；Odin1 可用 `--output` 指定新目录。`--help` 列出 bag、标定、两个权重、源码路径、帧数、设备、线程和投影参数。默认仅在 bag 与 `/workspace/dToF-DMDC/outputs/03/manifest.json` 匹配且参考帧数足够时自动使用该参考；此时 `--frames 1` 读取参考中的第一帧。新 bag 自动按原 DMDC 20%–80% 的规则独立选帧，不需要指定 `--reference`。可用 `--reference 目录` 显式比较对应 DMDC 结果（仍严格检查 bag 和帧数），或用 `--reference ''` 强制独立选帧。

独立的 `scripts/odin1_common.py` 保留了 DMDC 的 FishPoly 去畸变、外参变换、最近 Z 投影、距离和置信度过滤、分距离段统计及 PLY 导出函数。默认 800×648、水平 90°、置信度 ≥35、LiDAR 距离 ≥0.1 m、时间差 ≤50 ms。参考模式在运行时逐元素核对 RGB、稀疏深度、内外参；图像为 `[1,3,H,W]` RGB `[0,1]`，稀疏深度为 `[1,1,H,W]` 米制相机 Z、缺失值置零。模型内部依上游规则调整到 14 的倍数，再返回原始尺寸。

每帧保存 `rgb.png`、`sparse_depth.npy`、`depth.npy`、最终 `mask.png`、`model_mask.png`、`rectification_mask.png`、输入及预测彩色 PLY、深度预览及 `comparison.png`；目录根有 `overview.png`、`manifest.json` 和 `validation.json`。float32 NPY 是深度主产物，无效处为 NaN。最终有效区域为去畸变掩码、模型掩码、有限正深度的交集。PLY 使用已知相机内参反投影，坐标为 X 右、Y 下、Z 前，单位米；不使用模型估计的射线。

manifest 保存源提交、实际 LDCM diff、入口与核心源码 SHA256、两个权重 SHA256 和来源、依赖版本、精确时间戳、加载及逐帧推理耗时、有效比例及输入一致性分段统计。validation 检查保存的 NPY/掩码、PLY 顶点数、RGB、XYZ 反投影关系，以及与 DMDC 输入的逐元素一致性。比较图将两模型限制到共同有效区域，三帧和两模型统一使用合并预测的 1%–99% 米制色标；原始 NPY/PLY 不截断深度。

共同有效区域内还分别统计两模型与稀疏输入的一致性。它仅衡量对输入的拟合，不代表真实精度；模型间深度差也不是精度指标。DMDC 耗时来自已有运行，LDCM 来自此次运行，并非同进程基准测试。

## 迁移前的运行结果

以下结果及其原始 manifest 保留于 `/workspace/dToF-DMDC/outputs/ldcm_03` 和 `outputs/ldcm_sample_1`，不修改其中历史路径。

base CPU、四线程；官方 sample_1 推理 8.82 s，泊松补全及 640×480 预测导出检查通过。Odin1 完整模型加载耗时 2.39 s。

| 帧 | LDCM 推理 s | DMDC 已有推理 s | LDCM 覆盖率 | DMDC 覆盖率 | 共同有效像素 |
| --- | --- | --- | --- | --- | --- |
| 0 | 10.10 | 1.69 | 88.00% | 88.35% | 454037 |
| 1 | 10.59 | 1.63 | 99.84% | 99.87% | 517560 |
| 2 | 10.15 | 1.77 | 85.07% | 85.12% | 440747 |

三帧输入 RGB、稀疏深度、内外参与 DMDC 逐元素一致，头时间戳分别为：

| 帧 | RGB ns | cloud ns |
| --- | --- | --- |
| 0 | 522832171920 | 522831774762 |
| 1 | 545446434984 | 545445665951 |
| 2 | 568157797141 | 568157038733 |

在共同有效像素的稀疏输入位置，AbsRel（分段按像素数加权）如下。这是输入一致性，不是真实精度。

| 帧 | LDCM 输入 AbsRel | DMDC 输入 AbsRel |
| --- | --- | --- |
| 0 | 1.36% | 3.13% |
| 1 | 1.00% | 1.72% |
| 2 | 2.73% | 3.49% |

所有预测均为 648×800；三帧有效点数为 456186、517582、440979。二进制 PLY 的点数、颜色与相机内参反投影通过检查，最大 float32 舍入差约 1.91×10⁻⁶ m。详见 `outputs/ldcm_03/validation.json`。

## 新 bag 测试

```bash
cd /workspace/LDCM
/root/miniconda3/bin/python scripts/infer_odin1_ldcm.py \
  --bag /你的路径/新bag目录 \
  --calib /你的路径/calib.yaml \
  --output outputs/new_bag \
  --frames 3
```

独立选帧时，`--frames 1` 仅选择中间一帧；默认参考模式则使用参考中的第一帧。bag 必须包含 `/odin1/image/compressed` 和 `/odin1/cloud_raw`，标定须为 FishPoly。测试官方示例可用 `scripts/infer_ldcm_sample.py --output outputs/sample_test`。

## 迁移验证

从 `/tmp` 启动迁移后的官方 sample_1 和 Odin1 参考第一帧，分别输出到 `/tmp/ldcm_migrated_sample`、`/tmp/ldcm_migrated_odin1`。Odin1 第一帧预测与迁移前逐元素一致；sample_1 深度最大差异为 7.63×10⁻⁵ m（最大相对差异 3.73×10⁻⁵），在 `rtol=1e-4, atol=1e-5` 内一致，有限值掩码相同。Odin1 输入一致性、PLY 点数与反投影检查通过。数据处理函数的 AST 与原 DMDC 版本一致；新入口不导入 DMDC Python 模块。
