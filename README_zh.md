# Taming I2V models for Image HOI Editing: A Cognitive Benchmark and Agentic Self-Correcting Framework

这是 **HOI-Edit**、**HOI-Eval** 和 **SCPE** 的官方代码发布。

Current image editing methods excels at static attributes but fails at complex Human-Object Interactions (HOI), a critical challenge unaddressed by existing benchmarks that conflate HOI with static attributes, relying on global metrics incapable of simultaneously assessing dynamic interaction validity and entangled human-object pair preservation. Thus, we first introduce **HOI-Edit**, a comprehensive benchmark with three progressive cognitive levels, which features an automated metric **HOI-Eval** that first reliably evaluates instance-level interaction by letting VLM Q&A after thinking with images containing grounded Human-Object pair. Considering the task's essence of remodeling dynamic relationships, we benchmark Image-to-Video (I2V) models, finding them inherently suited for dynamic editing due to their temporal generation capabilities. Crucially, beyond superior performance, this capability provides a "replay of the failure process", offering unique diagnosability into why errors occur. We thus propose **SCPE (Self-Correcting Process Editing)**, a novel, agentic self-correcting framework that constrains the generation of I2V models through iteratively refined prompts, enabling the generated videos to more accurately present the target HOI. Extracted frames from these videos are the final editing results. On HOI-Edit, SCPE achieves performance competitive with state-of-the-art (SOTA) editing models like Nano Banana on interaction.

大文件资源单独下载，见 [ASSETS.md](ASSETS.md)。

## 下载资源

从 [ASSETS.md](ASSETS.md) 下载 `hoi_edit_assets_v7.tar.gz`，在仓库根目录解压：

```bash
tar -xzf hoi_edit_assets_v7.tar.gz
```

解压后结构：

```text
data/
├── collected_annotations_bboxes_v7_L1L2_questions_scoring_final.json
├── collected_annotations_bboxes_v7_L3_questions_scoring_final.json
├── data_v7_L12/
└── data_v7_L3/

weights/
├── groundingdino_swint_ogc.pth
└── sam2.1_hiera_large.pt
```

## SCPE 配置

SCPE 全称为 **Self-Correcting Process Editing**，代码位于 [scpe/](scpe)。它使用 Gemini 进行 Playbook learning、inference 和 frame selection，可通过 DashScope Wan2.2 I2V 或本地 Wan2.2 I2V A14B 生成视频。

```bash
cd scpe
./setup_env.sh
cp env.example env.local
```

编辑 `scpe/env.local`：

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export DASHSCOPE_API_KEY="your-dashscope-api-key"  # 仅 DashScope backend 需要
export ACE_LANG="cn"
export DATA_ROOT="/path/to/data"
export WAN22_REPO="/path/to/Wan2.2"
export WAN22_CKPT_DIR="/path/to/Wan2.2-I2V-A14B"
```

## SCPE 运行

1. 原 Wan2.2 生成过程：

```bash
cd scpe
source env.local
./run_minimal.sh wan22
```

2. SCPE 学习一轮 Playbook：

```bash
./run_minimal.sh learn
```

3. 用学完一轮的 Playbook 做 prompt enhancement 和抽帧：

```bash
./run_minimal.sh enhance
./run_minimal.sh wan22
./run_minimal.sh qa2
```

本地 Wan2.2：

```bash
WAN22_BACKEND=local ./run_minimal.sh wan22
```

## HOI-Eval

把编辑帧放在：

```text
data/<model_name>_frames/L1L2/
data/<model_name>_frames/L3/
```

运行：

```bash
MODELS=<model_name> bash run_eval.sh
```
