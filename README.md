# PC 侧训练项目框架

本目录用于在数据集未到位时，先把 PC 侧训练、验证、导出与推理的工程骨架搭起来，后续只需补数据和调参即可。

## 目录结构

```text
Project/
├── data/
│   ├── raw/                 # 原始数据占位
│   └── processed/           # 处理后的特征、切分结果、缓存文件
├── experiments/             # 实验输出目录（日志、权重、结果）
├── exports/                 # 导出模型目录（ONNX / TFLite 等）
├── scripts/                 # 训练、数据准备、导出入口脚本
├── src/
│   ├── configs/             # 训练 / 数据 / 模型配置
│   ├── data/                # 数据读取、预处理、增强
│   ├── models/              # TCN-BiLSTM 等模型定义
│   └── utils/               # 工具函数
├── requirements.txt
└── README.md
```

## 当前阶段建议

1. 先完成关键点序列数据格式定义
2. 用少量假数据跑通训练流程
3. 数据集到位后替换 `data/raw/` 并重新处理
4. 固化导出格式，优先 ONNX

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 创建数据目录

仓库已预留以下目录：

- `data/raw/`
- `data/processed/`
- `experiments/`
- `exports/`

### 3. 运行数据准备骨架

```bash
python scripts/prepare_data.py
```

### 4. 运行训练骨架

```bash
python scripts/train.py
```

## 配置说明

训练相关参数集中在 `src/configs/default.yaml`，主要包括：

- 项目名与随机种子
- 数据路径与序列长度
- 模型结构参数
- 训练超参数
- 导出参数

## 后续可扩展

- MediaPipe 关键点提取
- 时间窗滑动采样
- TCN-BiLSTM 分类
- 混淆矩阵与 Top-K 评估
- ONNX 导出与推理一致性检查
