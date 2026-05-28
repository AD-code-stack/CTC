# 孤立词手语识别项目

本项目用于在 PC 端完成**孤立词手语识别**的数据准备、特征提取、模型训练、评估与导出，后续再根据需要迁移到端侧或开发板推理。

当前项目目标是：

> **输入一段孤立词骨架序列，输出对应的词语类别。**

## 当前任务定义

这是一类标准的**样本级多分类任务**：

```text
原始骨架 txt
  -> 读取字典与样本扫描
  -> 统一序列长度
  -> 保存为 .npy 特征文件
  -> 生成 label map 与 train/val/test manifest
  -> PyTorch Dataset / DataLoader
  -> TCN-BiLSTM 分类器
  -> CrossEntropyLoss
  -> Accuracy / Macro F1 / Top-5 Accuracy
```

第一版优先使用骨架关键点序列作为输入特征，以降低训练和部署成本。

## 数据集说明

本项目当前使用的孤立词数据目录应放在仓库内的 `data/raw/` 下，当前配置默认读取：

- `data/raw/xf500_body_color_txt`：彩色图坐标系下的骨架 txt 序列，作为主输入
- `data/raw/xf500_body_depth_txt`：深度图坐标系下的骨架 txt 序列，可与 color 配对用于双模态融合
- `data/raw/dictionary.txt`：类别字典，负责把词语名称和类别编号对应起来

### 骨架数据格式

解压后，骨架 txt 文件通常表示：

- 一个文件对应一个样本
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
  -> 解压并放入 data/raw/
  -> 读取 dictionary.txt
  -> 扫描骨架 txt 文件
  -> 随机划分 train/val/test
  -> 统一序列长度
  -> 保存为 .npy 特征文件
  -> 生成 label map、manifest 与 summary
  -> PyTorch Dataset
  -> TCN-BiLSTM / 双分支 TCN-BiLSTM
  -> CrossEntropyLoss
  -> Accuracy / Macro F1 / Confusion Matrix
```

## 数据准备输出

第一阶段数据处理会生成以下产物：

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
- 所属 split

## 训练方法

当前训练支持以下几种模式：

- `single`：单分支 TCN-BiLSTM
- `dual`：双分支 color/depth 融合
- `auto`：自动根据数据模态选择

推荐优先使用：

- 输入：固定长度骨架序列
- 模型：`DualBranchTCNBiLSTM`
- 输出：整段样本的类别 logits
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

如果 `depth` 目录已经准备好，也可以显式指定：

```bash
python scripts/prepare_data.py --depth-dir data/raw/xf500_body_depth_txt
```

该步骤会扫描原始孤立词数据，提取骨架序列并生成可训练的特征文件与清单。

准备完成后脚本会自动做校验，包括：

- 三个 split 清单是否生成
- 特征文件是否存在
- 标签映射是否合法
- 数据统计文件是否存在

### 3. 后台训练

如果你想在服务器上后台运行训练，推荐使用：

```bash
nohup python -u scripts/train.py --fusion dual > train.log 2>&1 &
```

训练日志会写入 `train.log`，可以通过下面命令查看进度：

```bash
tail -f train.log
```

### 4. 网格搜索超参数

如果你想自动搜索更好的训练参数，可以使用网格搜索脚本。例如：

```bash
python scripts/grid_search.py \
  --grid train.lr=0.001,0.0005 \
  --grid train.weight_decay=0.0001,0.0005 \
  --grid model.dropout=0.2,0.3 \
  --grid train.batch_size=32
```

默认会把每次实验结果保存到 `experiments/grid_search/`。

### 5. 评估与推理

```bash
python scripts/eval.py
python scripts/infer.py --input path/to/sample.txt
```

如果需要导出 ONNX，可以继续使用：

```bash
python scripts/export_onnx.py
```

## 当前阶段说明

当前项目的重点是把孤立词识别这条链路先跑通，而不是连续手语识别。后续如果数据量和标注更加充分，再考虑扩展到：

- 双模态融合（color + depth）
- 更复杂的融合策略（双分支、门控、注意力）
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
- `src/data/slr_isolated.py`
- `src/data/dataset.py`
- `src/models/tcn_bilstm.py`
- `src/configs/default.yaml`

后续如果你要继续扩展，我可以再帮你补上：

- 更稳健的标签解析
- 分层随机划分
- 混淆矩阵导出
- ONNX 导出与推理示例
