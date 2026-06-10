# 孤立词手语识别项目

本项目用于在 PC 端完成**孤立词手语识别**的数据准备、特征提取、模型训练、评估与导出，后续再根据需要迁移到端侧或开发板推理。

当前项目目标是：

> **输入一段孤立词骨架序列，输出对应的词语类别。**

## 当前任务定义

这是一类标准的**样本级多分类任务**：

```text
输入骨架 txt
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

- `data/raw/isolated_word_hand_upper_txt`：由帧图像提取出的上半身 + 双手骨架 txt 序列，作为主输入
- `data/raw/dictionary.txt`：类别字典，负责把词语名称和类别编号对应起来

### 当前数据组织方式

这批数据按类别目录组织，目录名就是词典里的原始编号，例如：

- `data/raw/isolated_word_hand_upper_txt/000`
- `data/raw/isolated_word_hand_upper_txt/001`
- `data/raw/isolated_word_hand_upper_txt/002`

也就是说，原始目录编号、词典编号和最终训练标签会尽量保持一致，避免推理时发生标签错位。

### 新数据的来源

你目前使用的数据是由逐帧图像提取得到的骨架序列。其提取逻辑是：

- 输入：按样本组织的帧图像文件夹
- 通过 MediaPipe Holistic 提取：
  - 上半身关键点（肩、肘、腕）
  - 左手 21 个关键点
  - 右手 21 个关键点
- 每帧输出一行浮点数，保存为 `.txt`

每帧的理论特征维度是：

- 6 个上半身点 × 3 维 = 18 维
- 左手 21 点 × 3 维 = 63 维
- 右手 21 点 × 3 维 = 63 维

因此每帧理论上是 `18 + 63 + 63 = 144` 维；但由于不同帧可能存在缺失点或空值，实际读取时会做补零和对齐处理。

## 推荐的项目流程

```text
原始 SLR 孤立词帧图像
  -> MediaPipe Holistic 提取骨架 txt
  -> 扫描骨架 txt 文件
  -> 随机划分 train/val/test
  -> 统一序列长度
  -> 保存为 .npy 特征文件
  -> 生成 label map、manifest 与 summary
  -> PyTorch Dataset
  -> TCN-BiLSTM
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

当前训练支持以下模式：

- `single`：单分支 TCN-BiLSTM（当前新数据推荐）
- `auto`：自动根据数据模态选择

推荐优先使用：

- 输入：固定长度骨架序列
- 模型：`TCNBiLSTM`
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

如需指定别的数据源，可以通过配置文件修改 `data.raw_dir`。

该步骤会扫描帧图像或骨架文本，生成可训练的特征文件与清单。

准备完成后脚本会自动做校验，包括：

- 三个 split 清单是否生成
- 特征文件是否存在
- 标签映射是否合法
- 数据统计文件是否存在

### 3. 后台训练

如果你想在服务器上后台运行训练，推荐使用：

```bash
nohup python -u scripts/train.py > train.log 2>&1 &
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

如果你已经切换到新的数据集版本，建议重新生成 `data/processed/`，避免旧实验产物干扰当前结果。默认会把每次实验结果保存到 `experiments/grid_search/`。

### 5. 导出模型

训练完成后，可以使用导出脚本生成板端部署可用的模型文件：

```bash
python scripts/export_model.py \
  --checkpoint experiments/isolated_word/best_model.pt \
  --fusion single \
  --format both \
  --output-dir exports/single_best
```

导出结果包括：

- `model_ts.pt`：TorchScript 模型
- `model.onnx`：ONNX 模型
- `export_metadata.json`：导出元信息与标签映射

导出前请确保：

- 输入特征维度与训练时一致
- 序列长度与训练时一致
- 模型结构与 checkpoint 对应

### 6. 评估与推理

```bash
python scripts/eval.py
python scripts/infer.py --input path/to/sample.txt
```

## 当前阶段说明

当前项目的重点是把孤立词识别这条链路先跑通，并作为后续连续手语翻译的基础。后续如果数据量和标注更加充分，再考虑扩展到：

- 连续手语分割与词级识别
- 语言模型融合
- 更复杂的时序解码
- 更复杂的时序模型
- 板端实时推理与部署

## 连续手语翻译下一步计划

当前孤立词模型已经具备较好的基线，下一步建议按以下路线推进连续手语翻译：

### Phase 1：数据与任务定义
- 确定连续手语数据来源与格式
- 统一视频、骨架、词级/句级标注方式
- 设计训练/验证/测试划分
- 明确是否做词边界标注或直接做句子级翻译

### Phase 2：连续输入建模原型
- 复用当前骨架特征提取流程
- 将连续视频切分为滑动窗口序列
- 用当前 `TCNBiLSTM` 作为窗口级编码器
- 输出窗口级词候选或中间表示

### Phase 3：序列解码
- 在窗口级编码器上加入 CTC / Transformer 解码头
- 支持变长输出
- 初步完成连续词识别（Continuous Isolated Word Recognition）

### Phase 4：语言层建模
- 引入词序约束或语言模型
- 融合上下文信息，减少重复与跳词
- 提升句子级可读性

### Phase 5：端侧部署
- 导出 TorchScript / ONNX
- 验证板端推理性能
- 优化输入预处理、推理延迟与内存占用

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
- `scripts/export_model.py`
- `src/data/slr_isolated.py`
- `src/configs/default.yaml`

后续如果你要继续扩展，我可以再帮你补上：

- 更稳健的标签解析
- 分层随机划分
- 混淆矩阵导出
- ONNX 导出与推理示例
