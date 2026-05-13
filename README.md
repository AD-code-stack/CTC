# PC 侧训练项目框架

本目录用于在 PC 端先完成数据准备、特征提取、模型训练和导出，后续再把同样的特征形式迁移到开发板实时推理。

当前项目已经从“视频统计特征”切换为“**MediaPipe 手部关键点序列特征**”，并进一步升级为“**连续手语识别准备阶段**”：数据中的 `Gloss` 已按序列 token 化，训练输入与后续连续识别/翻译任务保持一致，便于后续扩展。

## 当前代码流程

整体流程如下：

```text
原始视频
  -> scripts/prepare_data.py
  -> MediaPipe 手部关键点提取
  -> 固定长度关键点序列（当前默认 32 帧 x 84 维）
  -> data/processed/features/*.npy
  -> scripts/train.py
  -> TCN-BiLSTM 编码器
  -> 帧级分类 logits
  -> CTC loss / greedy decode
  -> gloss 序列
  -> gloss-to-natural-language 转换器（后续）
  -> 自然语言输出
```

其中：

- `TCN-BiLSTM` 负责提取连续手语视频的时序特征，当前保留为第一版 baseline 主干
- `CTC` 用于在没有词级时间边界标注的情况下学习“视频帧序列 -> gloss token 序列”的对齐关系
- `gloss-to-natural-language 转换器` 是后续的第二阶段模块，用于把 `gloss` 序列转换为更自然的中文句子，可以先接规则后处理或大模型 API，再考虑训练专门转换器

## 当前进度

截至目前，项目已经完成了关键的数据与训练准备工作：

- 已完成 PC 端数据准备骨架
- 已完成基于 MediaPipe Hands 的关键点特征提取
- 已生成并验证首个样本特征文件，格式为 `(32, 84)`
- 已将 CE-CSL 的 `Gloss` 处理为 `token` 序列，为连续识别做准备
- 已完成 `TCN-BiLSTM` 模型骨架和配置对齐
- 已补充 README 中的流程说明和环境要求
- 已验证 `prepare_data.py` 可在当前环境中开始批量处理视频数据
- 已补充连续手语识别准备所需的 `token_map.json` 与 `sequence_prep_summary.json`
- 已明确后续可以在 `TCN-BiLSTM` 后接 `CTC / decoder`，再接 `gloss-to-natural-language` 转换器

当前仍待完善的部分：

- 更细粒度的评估分析（如按类别统计、混淆矩阵、Top-K 等）
- 更稳健的 checkpoint 恢复机制
- ONNX 导出脚本
- 开发板端实时推理与部署适配

## 训练过程与日志输出

当前 `scripts/train.py` 已补齐完整的训练主流程，包含：

- 训练集 / 验证集 / 测试集划分
- 每个 epoch 的训练与验证
- CTC loss 计算与反向传播
- 梯度裁剪
- 验证集指标评估
- 最优模型与最新模型保存
- 测试集最终评估

训练过程会额外保存便于后续作图分析的日志文件：

- `experiments/train_history.json`
- `experiments/train_history.csv`
- `experiments/final_metrics.json`

其中 `train_history.csv` 适合直接用 Excel、Pandas 或 Matplotlib 绘图，主要记录了：

- `train_loss`
- `train_token_error_rate`
- `train_token_accuracy`
- `train_edit_distance`
- `val_loss`
- `val_token_error_rate`
- `val_token_accuracy`
- `val_edit_distance`

如果后续需要继续增强训练过程，推荐优先补充：

- 学习率调度器
- checkpoint 断点续训
- 更完整的序列级评估脚本

### 当前全流程

如果原始数据已经放置完成，当前可执行的主流程为：

```bash
python scripts/prepare_data.py
python scripts/train.py
```

前者负责生成关键点特征与 `Gloss` 序列清单，后者负责完成连续标签准备并导出 `token_map` 与训练摘要。

### 连续识别与自然语言转换链条

当前项目后续的目标链条可以理解为：

```text
视频 -> 关键点序列 -> TCN-BiLSTM 编码器 -> CTC / decoder -> gloss 序列 -> gloss-to-natural-language 转换器 -> 自然语言
```

其中自然语言转换器有两种可行路线：

1. **先接大模型 API**：开发快、效果通常更自然，适合前期验证展示效果
2. **再训练专门转换器**：更可控、可离线部署、成本更低，但需要更多数据和训练工作



### 1. 数据准备

`scripts/prepare_data.py` 会读取 `data/raw/CE-CSL/` 中的视频和标签，按 `train / dev / test` 划分，逐个视频提取手部关键点序列。

### 2. 特征提取

每个视频会被均匀采样到固定长度（默认 32 帧），每帧使用 MediaPipe Hands 检测双手关键点：

- 左手 21 个点 × x/y
- 右手 21 个点 × x/y
- 合计 84 维

最终每个样本保存为一个 `.npy` 文件，形状通常是：

```text
[32, 84]
```

### 3. 训练

`scripts/train.py` 会：

- 读取配置 `src/configs/default.yaml`
- 设置随机种子
- 加载 `data/processed/ce_csl_manifest.json`
- 构建 `TCNBiLSTM`
- 打印数据集和模型信息

当前 `train.py` 仍是**训练骨架**，能验证数据与模型是否对齐，但还不是完整训练循环。后续如果你需要，我可以继续把完整训练、验证和保存权重补上。

## 目录结构

```text
project/
├── data/
│   ├── raw/                 # 原始数据放置目录
│   └── processed/           # 处理后的关键点特征、清单、统计信息
├── experiments/             # 训练日志、权重、结果
├── exports/                 # 导出模型目录（ONNX 等）
├── scripts/                 # 数据准备、训练、导出入口脚本
├── src/
│   ├── configs/             # 训练 / 数据 / 模型配置
│   ├── data/                # 数据集与特征提取逻辑
│   ├── models/              # TCN-BiLSTM 模型定义
│   └── utils/               # 工具函数
├── requirements.txt
└── README.md
```

## CE-CSL 数据处理流程

本项目采用“**原始视频 -> 关键点特征 -> 训练数据**”的方式，不直接端到端读视频做分类。

### 原始数据放置

将 CE-CSL 数据集放入：

```text
data/raw/CE-CSL/
├── label/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
└── video/
    ├── train/
    ├── dev/
    └── test/
```

每个 split 下再按 `Translator` 分成 `A` 到 `L` 子目录，视频文件放在对应目录内。

### 处理后输出

运行数据准备脚本后，会生成：

- `data/processed/features/train/*.npy`
- `data/processed/features/dev/*.npy`
- `data/processed/features/test/*.npy`
- `data/processed/ce_csl_manifest.json`
- `data/processed/ce_csl_summary.json`
- `data/processed/labels.json`

### 数据划分说明

CE-CSL 已经按照 `train`、`dev`、`test` 三部分划分好了，脚本不会重新随机切分，而是严格沿用原始划分。

## 运行方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

```bash
python scripts/prepare_data.py
```

如果原始数据路径正确，脚本会自动生成关键点特征文件和清单文件。

### 3. 运行训练骨架

```bash
python scripts/train.py
```

运行后会打印：

- 加载到的样本数
- 第一条样本的特征维度
- 模型参数量

这一步可以用来验证“数据 -> 模型”是否对齐。

### 4. 环境要求

当前关键点提取依赖 `MediaPipe Hands`，建议使用兼容版本：

```bash
pip install --force-reinstall mediapipe==0.10.14
```

如果当前环境中的 `mediapipe` 不带 `solutions` 接口，`prepare_data.py` 会直接报出明确错误，提示你重新安装兼容版本。

## 当前是否能跑通

### 能跑通的部分

- 读取配置
- 扫描原始视频和标签
- 使用 MediaPipe 提取手部关键点
- 保存处理后的 `.npy` 特征
- 加载 manifest 并构建 `TCN-BiLSTM`

### 还未完全实现的部分

- 完整训练循环（loss、backward、optimizer step）
- 验证集评估
- 最优模型保存
- ONNX 导出脚本
- 板端实时推理脚本

也就是说，**目前已经可以跑通“数据准备 + 模型构建”流程**，但还不是一个完整的训练工程。

## 配置说明

训练相关参数集中在 `src/configs/default.yaml`，主要包括：

- 项目名与随机种子
- 数据路径与序列长度
- 模型输入维度
- 模型结构参数
- 训练超参数
- 导出参数

## 后续可扩展

- 面部关键点提取
- 身体姿态关键点提取
- 双模态/多模态融合
- 完整训练与验证流程
- ONNX 导出与板端推理一致性检查
