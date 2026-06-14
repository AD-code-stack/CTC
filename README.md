# 孤立词手语识别项目

这是一个面向**孤立词手语识别**的端到端工程项目，当前已经完成了从原始帧图像到骨架序列、从骨架序列到可训练特征、从训练到模型导出的完整主链路。

项目当前的核心目标是：

> **输入一段孤立词骨架序列，输出对应的词语类别。**

当前阶段重点不是继续堆复杂结构，而是把这条链路稳定下来，为后续的板端部署系统设计做准备。

---

## 1. 当前项目进度概览

目前项目已经完成的工作如下。

### 已完成
- 帧图像到骨架关键点文本的提取流程
- 孤立词骨架数据的整理与清洗
- `data/raw/isolated_word_hand_upper_txt` 全量数据上传完成
- `000 ~ 499` 共 500 个类别目录已检查完毕
- 其中 `402` 类也已补齐到完整状态
- 数据准备脚本可扫描骨架 txt 并生成特征文件
- 训练脚本可基于生成的数据完成分类训练
- 标签映射已调整为尽量与 `dictionary.txt` 的原始编号保持一致
- 已完成全量 500 类训练并验证可稳定收敛
- 当前模型已切换为 `TCN + 全局平均池化 + 分类头`
- README 已统一为当前数据版本、训练状态与板端部署方向说明

### 当前已验证的实验结果
当前全量 500 类训练已经验证可稳定收敛，测试集结果达到较高水平，说明骨架特征配合当前轻量时序结构已经具备较强的分类能力，也说明这条 PC 端训练链路可以作为后续板端部署的基础。

### 当前阶段结论
基于已有结果，当前项目已经从“能不能跑通”进入到“如何工程化落地”的阶段。接下来更值得投入的方向是：

- 全量 500 类训练
- 导出模型
- 设计板端推理系统
- 做输入输出接口和实时推理流程
- 后续再决定是否继续做精调

---

## 2. 任务定义

当前任务是一个标准的**多分类识别任务**：

```text
骨架 txt 序列
  -> 统一长度
  -> 特征保存为 .npy
  -> 生成 label map / manifest / summary
  -> Dataset / DataLoader
  -> TCN + 全局平均池化 + 分类头
  -> CrossEntropyLoss
  -> Accuracy / Macro F1 / Top-5 Accuracy
```

这不是连续翻译任务，也不是检测任务，而是典型的**样本级分类**。

---

## 3. 当前数据集说明

### 数据目录
当前主数据目录为：

- `data/raw/isolated_word_hand_upper_txt`
- `data/raw/dictionary.txt`

其中：

- `isolated_word_hand_upper_txt` 是你上传的骨架 txt 数据
- `dictionary.txt` 是词典，用于把类别编号和词语名称对应起来

### 数据组织方式
当前数据按类别目录组织，例如：

- `data/raw/isolated_word_hand_upper_txt/000`
- `data/raw/isolated_word_hand_upper_txt/001`
- `data/raw/isolated_word_hand_upper_txt/002`
- ...
- `data/raw/isolated_word_hand_upper_txt/499`

这个结构的含义是：

- 目录名本身就是类别编号
- 目录编号和词典编号尽量保持一致
- 标签映射不再随意重排，避免推理时标签错位

### 完整性检查结果
目前数据检查结果如下：

- 类别总数：`500`
- 缺失类别：`0`
- 空类别：`0`
- `402` 类已补齐
- 当前可以直接切到全量数据训练

### 数据来源
这批数据来自逐帧图像提取的骨架序列。提取逻辑是：

- 输入：按样本组织的帧图像文件夹
- 使用 MediaPipe Holistic 提取：
  - 上半身关键点（肩、肘、腕）
  - 左手 21 个关键点
  - 右手 21 个关键点
- 每帧输出一行浮点数，保存为 `.txt`

### 单帧特征维度
理论上每帧特征维度为：

- 上半身 6 个点 × 3 维 = 18 维
- 左手 21 个点 × 3 维 = 63 维
- 右手 21 个点 × 3 维 = 63 维

合计：`144` 维/帧。

实际处理时会对缺失值做补零，并将序列统一对齐到固定长度。

---

## 4. 当前数据划分方式

当前训练集、验证集和测试集的划分方式是：

- **按类别分层随机划分**
- 每个类别内部先随机打乱
- 再按比例切分为：
  - `train`: 80%
  - `val`: 10%
  - `test`: 10%

这意味着：

- 不会把同一类别的样本全放到一个 split 里
- 各 split 的类别分布更均衡
- 对分类任务更合理

### 划分的确定性
- 使用固定随机种子
- 默认 seed 为 `42`
- 在数据不变的情况下，划分结果是稳定的

---

## 5. 数据处理产物

数据准备后会生成以下目录和文件：

- `data/processed/features/*.npy`
- `data/processed/manifests/train.json`
- `data/processed/manifests/val.json`
- `data/processed/manifests/test.json`
- `data/processed/labels/label_map.json`
- `data/processed/stats/dataset_summary.json`

### 各文件作用

#### `features/*.npy`
每个样本对应一个 `.npy` 特征文件，内容是统一长度后的骨架序列。

#### `manifests/*.json`
记录样本列表，包括：

- `sample_id`
- `feature_path`
- `label_id`
- `label_name`
- `source_path`
- `num_frames`
- `feature_dim`
- `split`

#### `label_map.json`
训练使用的标签映射表。
目前会尽量保持与 `dictionary.txt` 的原始编号顺序一致。

#### `dataset_summary.json`
记录本次数据准备的整体统计信息，包括：

- 样本总数
- 类别数
- 序列长度
- 特征维度
- 模态信息
- 划分比例
- 随机种子

---

## 6. 当前模型与训练状态

### 当前模型
当前训练使用的是：

- `TCN + 全局平均池化 + 分类头`

这是一个更轻量、更适合板端部署的结构，当前已经替代 `BiLSTM` 作为主模型，并且在全量 500 词训练中表现稳定。

### 当前训练方式
当前主流程是：

- 输入固定长度骨架序列
- 送入 `TCNBiLSTM`
- 输出整段样本的类别 logits
- 使用 `CrossEntropyLoss`

### 当前已验证的训练结果
当前全量 500 类模型在测试集上已经达到约 **96.8%** 的准确率、约 **96.8%** 的 Macro F1、约 **99.85%** 的 Top-5 Accuracy，说明当前方案具备较强的识别能力。

### 当前是否还需要继续调参
从现阶段看：

- 不是不能调
- 但已经可以先暂停大规模参数调优
- 更适合先进入模型导出和板端部署设计阶段

原因是当前结果已经很接近可用状态，继续调参的边际收益会变小。

---

## 7. 当前推荐工作流

现在推荐的工程流程是：

```text
原始帧图像
  -> MediaPipe Holistic 提取骨架 txt
  -> 统一目录结构
  -> 数据完整性检查
  -> 生成 .npy 特征文件
  -> 生成 train/val/test 清单
  -> 训练 TCN-BiLSTM
  -> 选择 best checkpoint
  -> 导出 TorchScript / ONNX
  -> 设计板端推理系统
```

---

## 8. 常用命令

### 8.1 准备数据

```bash
python scripts/prepare_data.py
```

### 8.2 启动训练

```bash
nohup python -u scripts/train.py > train.log 2>&1 &
```

查看日志：

```bash
tail -f train.log
```

### 8.3 导出模型

```bash
python scripts/export_model.py \
  --checkpoint experiments/isolated_word/best_model.pt \
  --fusion single \
  --format both \
  --output-dir exports/single_best
```

### 8.4 评估和推理

```bash
python scripts/eval.py
python scripts/infer.py --input path/to/sample.txt
```

---

## 9. 当前项目下一步建议

根据目前进度，下一步最合理的路线是：

### 第一优先级：全量 500 类训练
- 数据已经完整
- 标签映射已经整理
- 当前运行的就是全量 500 词训练

### 第二优先级：模型导出
- 导出 TorchScript / ONNX
- 为板端部署做准备
- 提前确认输入输出格式

### 第三优先级：板端系统设计
- 推理接口
- 数据预处理
- 输出解析
- 性能测试
- 低延迟优化

### 第四优先级：再做针对性调参
如果板端验证中出现：

- 延迟太高
- 个别类别混淆严重
- 需要更轻量模型

再回头做微调会更有针对性。

---

## 10. 连续手语翻译的后续路线

当前项目仍然以孤立词识别为主，但从工程上已经可以为连续手语翻译做铺垫。

### 后续路线建议

#### Phase 1：数据与任务定义
- 明确连续手语数据格式
- 确定视频、骨架、词级/句级标注方式
- 设计训练/验证/测试划分

#### Phase 2：连续输入原型
- 复用当前骨架提取流程
- 将连续视频切成滑动窗口
- 使用现有编码器先做窗口级表示

#### Phase 3：序列解码
- 引入 CTC 或 Transformer 解码头
- 支持变长输出
- 逐步完成连续词识别

#### Phase 4：语言层建模
- 引入上下文约束
- 融合语言模型
- 提升句子级输出稳定性

#### Phase 5：端侧部署
- 导出模型
- 验证板端推理性能
- 优化延迟、内存和输入输出链路

---

## 11. 关键文件

当前最重要的文件包括：

- `README.md`
- `scripts/prepare_data.py`
- `scripts/train.py`
- `scripts/export_model.py`
- `src/data/slr_isolated.py`
- `src/configs/default.yaml`

---

## 12. 总结

当前项目已经不再是“实验性原型”，而是进入了较完整的工程化阶段：

- 数据已完整
- 标签映射已整理
- 训练流程已验证
- 当前可以直接面向全量 500 类训练和模型导出

如果后续目标是板端部署，那么现在最重要的不是继续追逐小幅精度提升，而是把：

- 数据流程
- 模型导出
- 推理接口
- 系统架构

这几部分先稳定下来。
