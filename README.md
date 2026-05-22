# 孤立词手语识别项目

本项目用于在 PC 端完成**孤立词手语识别**的数据准备、特征提取、模型训练、评估与导出，后续再根据需要迁移到端侧或开发板推理。

当前数据集已切换到 `F:/Project/data/raw/SLR_Dataset/【孤立词】SLR_dataset`，项目目标也从连续手语识别调整为：

> **输入一段孤立词视频或骨架序列，输出对应的词语类别。**

## 当前任务定义

这是一类标准的**视频级多分类任务**：

```text
原始数据
  -> 数据解析与标签映射
  -> 骨架/关键点序列特征
  -> 固定长度序列张量
  -> 时序分类模型
  -> 类别预测
```

第一版优先使用骨架关键点序列作为输入特征，以降低训练和部署成本。

## 数据集说明

本项目当前使用的孤立词数据集目录中，重点需要的文件是：

- `dictionary.txt`：类别字典，负责把词语名称和类别编号对应起来
- `xf500_body_color_txt.zip`：彩色图坐标系下的骨架关键点序列，推荐作为第一版主输入
- `xf500_body_depth_txt.zip`：深度图坐标系下的骨架关键点序列，可作为后续增强或多模态输入

暂不作为第一版主输入的数据包括：

- RGB 视频分卷文件
- 连续句子识别数据
- CTC / gloss 序列标注链条

### 骨架数据格式

解压后，骨架 txt 文件通常表示：

- 一个文件对应一个样本视频
- 每行对应一帧
- 每行包含 50 个数值
- 表示 25 个关节点的 `x/y` 坐标

因此，每个样本可整理为一个二维序列：

```text
[T, 50]
```

其中 `T` 为帧数，后续会统一采样或插值到固定长度。

## 推荐的项目流程

```text
原始 SLR 孤立词数据
  -> 解压缩与目录扫描
  -> 读取 dictionary.txt
  -> 解析 body_color / body_depth 骨架序列
  -> 统一序列长度
  -> 保存为 .npy 特征文件
  -> 生成 label map 与 manifest
  -> PyTorch Dataset
  -> TCN-BiLSTM / CNN-LSTM 分类器
  -> CrossEntropyLoss
  -> Accuracy / Macro F1 / Confusion Matrix
```

## 当前版本的实现目标

项目第一版建议完成以下闭环：

- 读取 `dictionary.txt`
- 解析骨架 txt 数据
- 生成固定长度特征序列
- 构建分类训练集
- 训练孤立词分类模型
- 输出测试集评估结果
- 导出可推理模型

## 目录结构

```text
project/
├── data/
│   ├── raw/                 # 原始数据放置目录
│   └── processed/           # 处理后的特征、清单、标签映射、统计信息
├── experiments/             # 训练日志、权重、结果
├── exports/                 # 导出模型目录（ONNX 等）
├── scripts/                 # 数据准备、训练、评估、导出、推理入口脚本
├── src/
│   ├── configs/             # 配置文件
│   ├── data/                # 数据集与特征处理逻辑
│   ├── models/              # 分类模型定义
│   ├── training/            # 训练与评估逻辑
│   └── utils/               # 工具函数
├── requirements.txt
└── README.md
```

## 数据准备目标

第一阶段数据处理建议生成以下产物：

- `data/processed/features/*.npy`
- `data/processed/manifests/train.json`
- `data/processed/manifests/val.json`
- `data/processed/manifests/test.json`
- `data/processed/labels/label_map.json`
- `data/processed/stats/dataset_summary.json`

其中 `manifest` 记录每个样本的：

- 特征路径
- 标签 id
- 标签名称
- 样本来源信息

## 训练方法

第一版训练建议采用：

- 输入：固定长度骨架序列
- 模型：`TCN-BiLSTM` 或 `CNN-LSTM`
- 损失函数：`CrossEntropyLoss`
- 指标：`Top-1 Accuracy`、`Macro F1`、`Top-5 Accuracy`

### 训练流程

```text
特征文件 .npy
  -> Dataset / DataLoader
  -> 分类模型
  -> CrossEntropyLoss
  -> 反向传播
  -> 验证集评估
  -> 保存 best checkpoint
```

## 运行方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

```bash
python scripts/prepare_data.py
```

该步骤会扫描原始孤立词数据，提取骨架序列并生成可训练的特征文件与清单。

### 3. 开始训练

```bash
python scripts/train.py
```

训练脚本会读取处理后的数据、构建分类模型并输出训练日志。

### 4. 后续评估与导出

建议后续补充：

```bash
python scripts/eval.py
python scripts/export_onnx.py
python scripts/infer.py --video xxx.mp4
```

## 当前阶段说明

当前项目的重点是把孤立词识别这条链路先跑通，而不是连续手语识别。后续如果数据量和标注更加充分，再考虑扩展到：

- 双模态融合（color + depth）
- RGB 视频分类
- 更复杂的时序模型
- 端侧实时推理

## 配置说明

训练相关参数集中在 `src/configs/default.yaml`，主要包括：

- 项目名与随机种子
- 数据路径与序列长度
- 特征维度
- 模型输入维度
- 分类类别数
- 训练超参数
- 导出参数

## 目前最重要的文件

现在优先需要关注的文件是：

- `README.md`
- `scripts/prepare_data.py`
- `scripts/train.py`
- `src/data/feature_pipeline.py`
- `src/data/dataset.py`
- `src/configs/default.yaml`

后续我会继续把这些文件逐步改造成孤立词识别版本。
