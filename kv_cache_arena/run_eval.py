"""
KV Cache Arena — 评估入口
用法:
    python run_eval.py                # 完整评估
    python run_eval.py --quick        # 快速测试（前 50 条）
    python run_eval.py --no-baseline  # 只跑自己的策略，跳过基线
    python run_eval.py --budget 64    # 覆盖 strategy.py 中的 MAX_BUDGET
"""
import argparse
import sys
import os

# 让 import 找到同级文件
sys.path.insert(0, os.path.dirname(__file__))

import strategy
import engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',       action='store_true',
                        help='只跑前 50 条样本（快速验证）')
    parser.add_argument('--no-baseline', action='store_true',
                        help='跳过无压缩基线，只评估学生策略')
    parser.add_argument('--budget',      type=int, default=None,
                        help='覆盖 strategy.MAX_BUDGET')
    args = parser.parse_args()

    budget = args.budget if args.budget is not None else strategy.MAX_BUDGET

    engine.evaluate(
        compress_fn   = strategy.compress,
        max_budget    = budget,
        strategy_name = getattr(strategy, 'STRATEGY_NAME', 'MyStrategy'),
        n_docs        = 5 if args.quick else None,
        show_baseline = not args.no_baseline,
    )


if __name__ == '__main__':
    main()
