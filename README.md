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
  -> GatedFusionTCNBiLSTM
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
  -> GatedFusionTCNBiLSTM / 双分支 TCNBiLSTM
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
- `gated`：门控融合，当前推荐主模型
- `auto`：自动根据数据模态选择

推荐优先使用：

- 输入：固定长度骨架序列
- 模型：`GatedFusionTCNBiLSTM`
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
nohup python -u scripts/train.py --fusion gated > train.log 2>&1 &
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

### 5. 导出模型

训练完成后，可以使用导出脚本生成板端部署可用的模型文件：

```bash
python scripts/export_model.py \
  --checkpoint experiments/isolated_word/best_model.pt \
  --fusion gated \
  --format both \
  --output-dir exports/gated_best
```

导出结果包括：

- `model_ts.pt`：TorchScript 模型
- `model.onnx`：ONNX 模型
- `export_metadata.json`：导出元信息与标签映射

导出前请确保：

- 输入特征维度与训练时一致
- 序列长度与训练时一致
- 模型结构与 checkpoint 对应

## 板端部署参考

当前项目的最终落点是开发板或边缘端部署，因此建议优先保留以下链路：

```text
骨架 txt / 采集模块
  -> 统一预处理
  -> GatedFusionTCNBiLSTM
  -> TorchScript / ONNX 导出
  -> 板端推理
```

建议部署前检查：

- 输入张量 dtype 是否为 `float32`
- 输入 shape 是否与训练一致
- 标签映射是否和模型一致
- 导出后推理结果是否与 PyTorch 版本一致

## 连续手语翻译的参考实现思路

仓库外部已有一套连续识别流程可作为参考。其核心思路不是直接做句子级端到端翻译，而是：

```text
摄像头采集连续动作
  -> 感知模块转骨架序列
  -> 连续动作中抽取关键帧
  -> 不足则插值补齐
  -> 固定长度序列输入模型
  -> 输出单个手语词
  -> 前端再将多个词串联成句子
```

这条路线对当前项目非常有参考价值，因为它说明了一个现实可行的方向：

- 先做词级实时识别
- 再做边界切分
- 再把词串成句子

### 关键启发

该流程里，连续动作的处理重点包括：

- 对相邻帧做差，选出变化最大的关键帧
- 用关键帧组成固定长度输入
- 帧数不足时使用线性插值补齐
- 模型输出最后一个时间步的分类结果
- 使用置信度阈值辅助稳定输出

这意味着当前孤立词模型可以作为连续手语翻译的基础模块，后续最可行的扩展路线是：

1. 实时窗口化采集连续骨架
2. 用边界动作或静止段做切分
3. 对每段调用现有孤立词模型
4. 再在前端或语言层做句子拼接

## 当前阶段说明

当前项目的重点是把孤立词识别这条链路先跑通，并作为后续连续手语翻译的基础。后续如果数据量和标注更加充分，再考虑扩展到：

- 连续手语分割与词级识别
- 语言模型融合
- 更复杂的时序解码
- RGB / depth / skeleton 多模态协同
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
- 用当前 `GatedFusionTCNBiLSTM` 作为窗口级编码器
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
- `src/data/dataset.py`
- `src/models/tcn_bilstm.py`
- `src/configs/default.yaml`

后续如果你要继续扩展，我可以再帮你补上：

- 更稳健的标签解析
- 分层随机划分
- 混淆矩阵导出
- ONNX 导出与推理示例
