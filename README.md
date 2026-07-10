# MusicTranscription

**Music Transcription and Generation**

## 简介 / Introduction

本项目提出了一种从短多轨录音生成音乐的新范式。为克服处理原始音频波形的复杂性和高数据量挑战，我们将音乐特征转换为 MIDI 编码格式，再利用生成模型扩展为更长的多轨音乐作品。整体流程分为两个阶段：

1. **音乐转录**（Audio → MIDI）：使用深度学习模型从频谱图中提取音乐特征，将音频文件转录为 MIDI 文件
2. **音乐生成**（MIDI → Extended Music）：基于改进的 Sparse Transformer 模型，从 MIDI 输入生成更长的多轨音乐

This project proposes a novel paradigm for generating music from short multi-track recordings. To overcome the complexity of processing raw audio waveforms and the high data volume, we transform musical features into a MIDI-encoded format and then use a generation model to produce extended multi-track compositions. The overall pipeline consists of two stages:

1. **Music Transcription** (Audio → MIDI): Extract musical features from spectrograms using deep learning models and transcribe audio into MIDI files
2. **Music Generation** (MIDI → Extended Music): Generate longer multi-track music from MIDI input using an improved Sparse Transformer model

> **代码范围**：本仓库包含转录阶段（T5、Albert）的完整实现；RNN-CNN 转录模型与生成阶段的代码未并入本仓，相关方法详见 `report/`。
>
> **Code scope**: This repository contains only the transcription stage (T5, Albert). The RNN-CNN transcription model and the generation stage are described in `report/`, but their code is not merged into this repo.

### 转录模型 / Transcription Models

我们实现并比较了三种转录模型：

- **RNN-CNN**：结合卷积神经网络和循环神经网络的混合架构
- **T5**：基于 [MT3](https://arxiv.org/abs/2111.03017) 的编码器-解码器 Transformer，将转录任务视为翻译任务（onset F1: 0.9712）
- **Albert**：基于 Albert 的 token 分类模型（12 层隐藏层，输入长度 512），将转录任务视为逐帧分类任务

We implemented and compared three transcription models:

- **RNN-CNN**: A hybrid architecture combining convolutional and recurrent neural networks
- **T5**: An encoder-decoder Transformer based on [MT3](https://arxiv.org/abs/2111.03017), treating transcription as a translation task (onset F1: 0.9712)
- **Albert**: An Albert-based token classification model (12 hidden layers, input length 512), treating transcription as a frame-level classification task

### 生成模型 / Generation Model

在生成阶段，我们对 [SymphonyNet](https://arxiv.org/abs/2205.05448) 进行了以下改进：

- 将线性注意力替换为 **Sparse Transformer** 注意力机制，显著提升收敛速度（~5k steps vs ~40k steps）
- 引入 **Rotary Position Embedding (RoPE)** 替代相对位置编码，更灵活地捕获序列内关系
- 统一训练和推理阶段的架构，采用标准自回归方式生成，避免训练/推理不一致的问题

实验表明，与 Google Magenta 的 Continue 工具相比，我们的模型在多轨音乐生成的复杂度和质量上表现更优。

For the generation stage, we made the following improvements over [SymphonyNet](https://arxiv.org/abs/2205.05448):

- Replaced linear attention with **Sparse Transformer** attention, significantly improving convergence speed (~5k steps vs ~40k steps)
- Introduced **Rotary Position Embedding (RoPE)** to replace relative position encoding for more flexible sequence modeling
- Unified the architecture between training and inference phases, using standard autoregressive generation to avoid train/inference inconsistency

Experiments show that our model outperforms Google Magenta's Continue tool in the complexity and quality of multi-track music generation.

## 项目结构 / Project Structure

```
MusicTranscription/
├── train.py              # 训练脚本 / Training script
├── inference.py          # 推理脚本 / Inference script
├── dataset.py            # 数据集处理 / Dataset processing
├── utils.py              # 工具函数 / Utility functions
├── config.yaml           # 训练配置 / Training configuration
├── config/               # 模型配置 / Model configurations
│   ├── T5.json
│   └── Albert.json
├── models/               # 模型实现 / Model implementations
│   ├── t5.py
│   └── albert.py
├── contrib/              # 辅助模块 / Helper modules
│   ├── event_codec.py
│   ├── note_sequences.py
│   ├── preprocessor.py
│   ├── spectrograms.py
│   ├── vocabularies.py
│   └── ...
├── pretrain/             # 预训练权重（不含在仓库中）/ Pretrained weights (not in repo)
├── data/                 # 示例音频（训练集见下方 Datasets）/ Demo audio (see Datasets below)
├── results/              # 示例转录结果 / Demo transcription results
├── report/               # 研究报告 / Research report
│   ├── report.tex
│   └── Figure/
└── test.ipynb            # 测试 notebook / Test notebook
```

## 预训练模型 / Pretrained Models

以下预训练权重文件未包含在仓库中（体积过大），可通过以下方式获取：

The following pretrained weights are not included in the repository due to their large size. They can be obtained as follows:

| 文件 / File | 大小 / Size | 说明 / Description | 获取方式 / How to Obtain |
|---|---|---|---|
| `pretrain/mt3.pth` | ~176MB | T5-based MT3 转录模型 / T5-based MT3 transcription model | 使用 `train.py` 在 MAESTRO 数据集上训练，或从 [magenta/mt3](https://github.com/magenta/mt3) 获取官方权重并转换为 PyTorch 格式 / Train with `train.py` on [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset, or convert official weights from [magenta/mt3](https://github.com/magenta/mt3) |
| `pretrain/Albert.ckpt` | ~99MB | Albert 转录模型 / Albert transcription model | 使用 `train.py --model Albert` 在 MAESTRO 数据集上训练 / Train with `train.py --model Albert` on [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset |

将权重文件放入 `pretrain/` 目录后即可使用。

Place the weight files in the `pretrain/` directory before use.

## 使用方法 / Usage

### 训练 / Training

```bash
# 训练 T5 模型 / Train T5 model
python train.py --model T5

# 训练 Albert 模型 / Train Albert model
python train.py --model Albert
```

### 推理 / Inference

```bash
python inference.py
```

推理脚本会从 `pretrain/` 加载预训练权重，将音频文件转录为 MIDI 文件。

The inference script loads pretrained weights from `pretrain/` and transcribes audio files into MIDI files.

## 数据集 / Datasets

- **转录 / Transcription**: [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) — 约 200 小时钢琴音频与 MIDI 的对齐数据集 / ~200 hours of aligned piano audio and MIDI recordings
- **生成 / Generation**: [Symphony Dataset](https://arxiv.org/abs/2205.05448) — 超过 46,000 首交响乐 MIDI 文件 / 46,000+ symphonic music MIDI files

## 参考文献 / References

- Gardner et al., "[MT3: Multi-Task Multitrack Music Transcription](https://arxiv.org/abs/2111.03017)", 2021
- Liu et al., "[Symphony Generation with Permutation Invariant Language Model](https://arxiv.org/abs/2205.05448)" (SymphonyNet), 2022
- Hawthorne et al., "[Sequence-to-Sequence Piano Transcription with Transformers](https://arxiv.org/abs/2107.09142)", 2021
- Child et al., "[Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)", 2019

## 声明

本项目为清华大学龙明盛老师《深度学习》课程研究项目，仅供学习交流参考，请勿直接复制用于课程作业提交。
