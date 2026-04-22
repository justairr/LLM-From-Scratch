# LLM From Scratch

一个从零手写大语言模型核心组件的动手实验 Notebook，适合希望深入理解 Transformer 内部机制的学习者。

A hands-on notebook for building the core components of a Large Language Model from the ground up. This project is designed for learners who want to dive deep into the internal mechanisms of the Transformer architecture.

## 内容概览

| 章节 | 主题 | 核心知识点 |
|------|------|-----------|
| Chapter 1 | 数据流转与分词 | 字符分词、滑动窗口、Token/Positional Embedding |
| Chapter 2 | 注意力机制 | 缩放点积注意力、因果掩码、多头注意力 (MHA) |
| Chapter 3 | 现代大模型骨架 | RMSNorm、SwiGLU、残差连接、Pre-Norm |
| Chapter 4 | 预训练逻辑 | Weight Tying、交叉熵 Loss、AdamW、梯度裁剪 |

最终组装出一个 ~100K 参数的 MiniLLM，并完成完整的训练循环。



## 环境配置
有个pytorch和ipykernel就能跑。最简环境可以参考下面。

```bash
conda create -n lfs python=3.10
conda activate lfs
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jupyter
```

## 快速开始

在llm-from-scratch.ipynb中按顺序执行每个 Cell，填写标注了 `[学生填空点]` 的代码块，并通过末尾的验证函数确认实现正确。

## 写在后面
本项目用于 HUST 自然语言处理课程实验，旨在一步步熟悉大模型本身。部分逻辑对于不熟悉python的同学可能较为复杂，所以在`validation_files/llm-from-scratch-answer.ipynb`中提供了一组参考解。

项目本身可能有各种BUG, 也欢迎大家提出一些其他的实验设计建议。