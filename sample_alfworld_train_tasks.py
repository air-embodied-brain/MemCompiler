#!/usr/bin/env python3
"""从 ALFWorld train 任务列表中按固定种子随机采样指定比例。"""
import argparse
import json
import os
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Sample ALFWorld train tasks by ratio with fixed seed.')
    parser.add_argument('--input_file', type=str, default='data/alfworld/alfworld_tasks_train.json',
                        help='Path to full train task json file.')
    parser.add_argument('--sample_ratio', type=float, default=0.4,
                        help='Sampling ratio in (0,1], e.g., 0.4 means 40%%.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--output_file', type=str, default='',
                        help='Output json path. If empty, auto-generate in same directory.')
    return parser.parse_args()


def build_output_path(input_file: str, sample_ratio: float, seed: int, output_file: str) -> str:
    if output_file:
        return output_file
    input_path = Path(input_file)
    ratio_tag = int(sample_ratio * 100)
    return str(input_path.parent / f"alfworld_tasks_train_sampled_{ratio_tag}_seed{seed}.json")


def main():
    args = parse_args()

    if not (0 < args.sample_ratio <= 1):
        raise ValueError(f"sample_ratio must be in (0, 1], got {args.sample_ratio}")

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    with open(args.input_file, 'r') as f:
        all_tasks = json.load(f)

    total_tasks = len(all_tasks)
    if total_tasks == 0:
        raise ValueError(f"Input file has 0 tasks: {args.input_file}")

    random.seed(args.seed)
    all_indices = list(range(total_tasks))
    sample_count = max(1, int(total_tasks * args.sample_ratio))
    sampled_indices = sorted(random.sample(all_indices, sample_count))

    sampled_tasks = [all_tasks[i] for i in sampled_indices]

    output_file = build_output_path(args.input_file, args.sample_ratio, args.seed, args.output_file)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(sampled_tasks, f, indent=2)

    # Save sampled indices next to sampled task file
    indices_file = os.path.join(os.path.dirname(output_file), 'sampled_indices.json')
    with open(indices_file, 'w') as f:
        json.dump({
            'seed': args.seed,
            'sample_ratio': args.sample_ratio,
            'total_tasks': total_tasks,
            'sample_count': sample_count,
            'indices': sampled_indices,
            'sampled_task_file': output_file
        }, f, indent=2)

    print(f"Total tasks in train set: {total_tasks}")
    print(f"Sampled {sample_count} tasks ({args.sample_ratio * 100:.0f}%) with seed {args.seed}")
    print(f"Sampled indices (first 10): {sampled_indices[:10]}")
    print(f"Saved sampled tasks to: {output_file}")
    print(f"Saved sampled indices to: {indices_file}")


if __name__ == '__main__':
    main()

