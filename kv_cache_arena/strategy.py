"""
KV Cache 压缩策略 — 学生实现文件
=====================================
只需修改这个文件，不要改动其他文件！

任务说明
--------
模型在推理时会积累 KV cache。cache 越大，内存占用越高。
你的任务是实现一个压缩函数，在 cache 超过 max_budget 时将它裁剪。
目标：在尽可能低的内存预算下，最大化模型在测试集上的准确率。

接口说明
--------
compress(past_key_values, max_budget, step) -> past_key_values

    past_key_values : 当前的 KV cache（transformers DynamicCache 对象）
    max_budget      : 最大允许保留的 token 数（由 MAX_BUDGET 控制）
    step            : 当前是第几个 token（0-indexed），可用于判断压缩时机

    返回值 : 压缩后的 past_key_values（直接修改后返回即可）

可用工具函数（来自 kv_utils）
------------------------------
    kv_utils.get_seq_len(pkv)                          -> int      当前 cache 长度
    kv_utils.get_device(pkv)                           -> device
    kv_utils.select_indices(pkv, indices)              -> pkv      按位置保留
    kv_utils.keep_recent(pkv, n)                       -> pkv      保留最近 n 个
    kv_utils.keep_random(pkv, n)                       -> pkv      随机保留 n 个

评分方式
--------
在 WikiText-2 测试集的长文本段落（200-512 tokens）上计算 Perplexity（困惑度）。
- Perplexity 越低越好（越接近无压缩基线越好）
- 退化率 ≤ 5%  ✅   退化率 ≤ 20%  ⚠️   退化率 > 20%  ❌

运行方式
--------
    python run_eval.py              # 完整评估（含基线对比）
    python run_eval.py --quick      # 快速测试（前 50 条）
    python run_eval.py --no-baseline  # 跳过基线，只跑自己的策略
"""

import torch
import kv_utils

# ============================================================
STRATEGY_NAME = 'MyStrategy' 
MAX_BUDGET    = 64            # cache 保留的最大 token 数（上下文约 200-400 tokens）
# ============================================================


def compress(past_key_values, max_budget: int, step: int):
    """
    在这里实现你的 KV cache 压缩策略。

    当 cache 中的 token 数超过 max_budget 时，此函数会被自动调用。

    参数:
        past_key_values : 当前的 KV cache
        max_budget      : 最多保留多少个 token
        step            : 当前是第几个 token（0-indexed）

    返回:
        压缩后的 past_key_values
    """
    # ============================================================
    # ✏️  在这里写你的实现
    # ============================================================

    # 策略 0：不压缩（满 cache，仅用于调试）
    # return past_key_values

    # 策略 1：FIFO —— 只保留最近的 token
    # return kv_utils.keep_recent(past_key_values, max_budget)

    # 策略 2：随机采样
    return kv_utils.keep_random(past_key_values, max_budget)
