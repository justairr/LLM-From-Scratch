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

在完成本实验后，你应该对模型结构及KV Cache有相关了解，此时请步入 `/kv_cache_areana` 体验kv cache压缩的内容。
在这里，我们将对一个真实模型`gpt-2`进行kv cache压缩，并尝试使用不同的压缩方法。

## 环境配置
有个pytorch和ipykernel就能跑。最简环境可以参考下面。

```bash
conda create -n lfs python=3.10
conda activate lfs
pip install torch --index-url https://download.pytorch.org/whl/cpu # 如果你有gpu最好用gpu
pip install ipykernel
```

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
cd /kv_cache_arena

# 快速验证（5 篇文章，约 1 分钟）
python run_eval.py --quick

# 完整评估（20 篇，含无压缩基线对比，约 4 分钟）
python run_eval.py

# 只跑自己的策略，不跑基线（节省一半时间）
python run_eval.py --no-baseline

# 临时修改 budget（不改 strategy.py）
python run_eval.py --budget 64
```

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
本项目用于 HUST 自然语言处理课程实验，旨在一步步熟悉大模型本身。部分逻辑对于不熟悉python的同学可能较为复杂，所以在`validation_files/llm-from-scratch-answer.ipynb`中提供了一组参考解。

项目本身可能有各种BUG, 也欢迎大家提出一些其他的实验设计建议。

# requirements
annotated-doc==0.0.4
anyio==4.13.0
asttokens==3.0.1
certifi==2026.4.22
click==8.3.3
comm==0.2.3
contourpy==1.3.2
cuda-bindings==13.2.0
cuda-pathfinder==1.5.4
cuda-toolkit==13.0.2
cycler==0.12.1
debugpy==1.8.20
decorator==5.2.1
exceptiongroup==1.3.1
executing==2.2.1
filelock==3.29.0
fonttools==4.62.1
fsspec==2026.3.0
h11==0.16.0
hf-xet==1.4.3
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.12.2
idna==3.13
ipykernel==7.2.0
ipython==8.39.0
jedi==0.19.2
Jinja2==3.1.6
jupyter_client==8.8.0
jupyter_core==5.9.1
kiwisolver==1.5.0
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.9
matplotlib-inline==0.2.1
mdurl==0.1.2
mpmath==1.3.0
nest-asyncio==1.6.0
networkx==3.4.2
numpy==2.2.6
nvidia-cublas==13.1.0.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.19.0.56
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.0
nvidia-nccl-cu13==2.28.9
nvidia-nvjitlink==13.0.88
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
packaging==26.0
pandas==2.3.3
parso==0.8.6
pexpect==4.9.0
pillow==12.2.0
platformdirs==4.9.6
prompt_toolkit==3.0.52
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.3
Pygments==2.20.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytz==2026.1.post1
PyYAML==6.0.3
pyzmq==27.1.0
regex==2026.4.4
rich==15.0.0
safetensors==0.7.0
seaborn==0.13.2
shellingham==1.5.4
six==1.17.0
stack-data==0.6.3
sympy==1.14.0
tokenizers==0.22.2
torch==2.11.0
tornado==6.5.5
tqdm==4.67.3
traitlets==5.14.3
transformers==5.7.0
triton==3.6.0
typer==0.25.0
typing_extensions==4.15.0
tzdata==2026.2
wcwidth==0.6.0
