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
| Chapter 5 | KV Cache | KV Cache的主要原理 |


最终组装出一个 ~100K 参数的 MiniLLM，并完成完整的训练循环。

---

在完成本实验后，你应该对模型结构及KV Cache有相关了解，此时请步入 `/kv_cache_arena` 体验 kv cache 压缩的内容。
在这里，我们将对一个真实模型`gpt-2`进行kv cache压缩，并尝试使用不同的压缩方法。

## 环境配置
有个pytorch和ipykernel就能跑。最简环境可以参考下面。

```bash
conda create -n lfs python=3.10
conda activate lfs
pip install torch --index-url https://download.pytorch.org/whl/cpu # 如果你有gpu最好用gpu
pip install ipykernel
```

`kv_cache_arena` 另外提供了一个单独的最小运行依赖文件：[`kv_cache_arena/requirements.txt`](kv_cache_arena/requirements.txt)。
如果你没有 GPU，建议先安装 CPU 版 PyTorch，再进入该目录执行 `pip install -r requirements.txt`。

## 快速开始

在llm-from-scratch.ipynb中按顺序执行每个 Cell，填写标注了 `[学生填空点]` 的代码块，并通过末尾的验证函数确认实现正确。



# KV Cache Arena

## 实验目标

大模型推理时，KV Cache 随上下文长度线性增长，耗尽显存是实际部署的主要瓶颈之一。
本实验要求你设计一个 **KV Cache 压缩策略**，在受限的 cache 预算下，尽量减少模型质量损失。

**评估指标**：在 WikiText-2 测试集的长文本段落上计算 **Perplexity（困惑度）**。
Perplexity 越低 = 压缩带来的质量损失越小。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `strategy.py` | ✏️ **你只需要修改这个文件** |
| `run_eval.py` | 评估入口，运行这个脚本看结果 |
| `kv_utils.py` | 工具函数（可调用，请勿修改） |
| `engine.py`   | 评估引擎（请勿修改） |

---

## 快速开始

```bash
# 进入实验目录
cd kv_cache_arena

# 安装最小运行环境（无 GPU 推荐先装 CPU 版 PyTorch）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 快速验证
python run_eval.py --quick

# 完整评估
python run_eval.py

# 只跑自己的策略，不跑基线（节省一半时间）
python run_eval.py --no-baseline

# 临时修改 budget（不改 strategy.py）
python run_eval.py --budget 64
```

如果你已经装好了合适版本的 GPU 版 PyTorch，可以跳过上面的 CPU 安装命令，只执行 `pip install -r requirements.txt`。
未安装 `datasets` 时，评测会自动退回到内置语料，仍然可以运行，但结果不再是标准 WikiText-2 测试集分数。

---

## 如何实现压缩策略

打开 `strategy.py`，修改 `compress()` 函数。

### 接口说明

```python
def compress(past_key_values, max_budget: int, step: int):
    """
    past_key_values : 当前 KV cache（transformers DynamicCache）
    max_budget      : 最多保留多少 token
    step            : 当前是第几个 token（0-indexed）
    返回值          : 压缩后的 past_key_values
    """
```

每当 cache 中的 token 数超过 `MAX_BUDGET`，引擎会自动调用你的 `compress()` 函数。

### 可用的工具函数（`kv_utils`）

```python
import kv_utils

kv_utils.get_seq_len(pkv)                          # 当前 cache 长度（token 数）
kv_utils.get_device(pkv)                           # cache 所在设备

# 按位置选择要保留的 token
kv_utils.select_indices(pkv, indices)              # indices: LongTensor

# 预置策略
kv_utils.keep_recent(pkv, n)                       # 保留最近 n 个（FIFO）
kv_utils.keep_random(pkv, n)                       # 随机保留 n 个
```

---

## 参考策略（从简单到复杂）

### 策略 1：FIFO（先进先出）
只保留最近的 token，丢弃最旧的。

```python
def compress(past_key_values, max_budget, step):
    return kv_utils.keep_recent(past_key_values, max_budget)
```

### 策略 2：自定义
利用 `step`、位置信息或其他启发式规则设计你自己的方案。

```python
def compress(past_key_values, max_budget, step):
    seq_len = kv_utils.get_seq_len(past_key_values)
    dev     = kv_utils.get_device(past_key_values)
    # 根据 step 或位置决定保留哪些 token
    indices = ...  # LongTensor，值域 [0, seq_len-1]
    return kv_utils.select_indices(past_key_values, indices)
```



# 写在后面
本项目用于 HUST 自然语言处理课程实验，旨在一步步熟悉大模型本身。
项目本身可能有各种BUG, 也欢迎大家提出一些其他的实验设计建议。

Thanks for [@Seas0](https://github.com/Seas0) for jailbreak testing.