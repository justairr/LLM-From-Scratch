# KV Cache Arena — 学生实验

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
cd llm-experiments/kv_cache_arena

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
