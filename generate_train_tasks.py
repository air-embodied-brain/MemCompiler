#!/usr/bin/env python3
"""生成 ALFWorld train 数据集任务列表（真实 reset 获取初始 observation）"""
import json
import os
from pathlib import Path

import yaml
import alfworld.agents.environment

TRAIN_DATA_PATH = '/path/to/alfworld/json_2.1.1/train'
OUTPUT_JSON = 'data/alfworld/alfworld_tasks_train.json'
ALFWORLD_CONFIG_PATH = 'tasks/env_configs/alfworld_config.yaml'


def _extract_init_obs(reset_output) -> str:
    """兼容解析 env.reset() 返回值，提取第一条 observation 文本。"""
    data = reset_output

    if isinstance(data, tuple):
        data = data[0] if len(data) > 0 else ''

    obs = ''
    if isinstance(data, str):
        obs = data
    else:
        try:
            obs = data[0] if len(data) > 0 else ''
        except Exception:
            obs = data

    if isinstance(obs, bytes):
        obs = obs.decode('utf-8', errors='ignore')

    obs = '' if obs is None else str(obs)
    return obs.strip()


def _split_room_and_task(obs_text: str) -> tuple[str, str]:
    """从 reset observation 中提取房间描述和任务描述。"""
    text = (obs_text or '').replace('\r\n', '\n').strip()

    # 去掉欢迎语横幅
    lines = [line for line in text.split('\n') if line.strip() != '']
    if lines and lines[0].startswith('-= Welcome to TextWorld'):
        lines = lines[1:]
    text = '\n'.join(lines).strip()

    # 兼容项目已有处理：去掉 "You arrive at loc ..."
    if text.startswith('You arrive at loc '):
        dot_idx = text.find('. ')
        if dot_idx != -1:
            text = text[dot_idx + 2:]

    marker = 'Your task is to:'
    if marker in text:
        room_desc, task_tail = text.split(marker, 1)
        room_desc = room_desc.strip()
        task_desc = task_tail.strip().split('\n')[0].strip().rstrip('.')
    else:
        room_desc = text.strip()
        task_desc = ''

    return room_desc, task_desc


def _build_alfworld_env(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['split'] = 'train'
    env_type = config['env']['type']
    return getattr(alfworld.agents.environment, env_type)(config, train_eval='train')


def collect_train_tasks(train_path: str):
    tasks = []
    task_id = 0
    skipped = 0

    train_dir = Path(train_path)
    if not train_dir.exists():
        print(f"错误：路径不存在 {train_path}")
        return tasks

    main_env = _build_alfworld_env(ALFWORLD_CONFIG_PATH)

    for task_type_dir in sorted(train_dir.iterdir()):
        if not task_type_dir.is_dir():
            continue
        print(f"处理任务类型: {task_type_dir.name}")

        for trial_dir in sorted(task_type_dir.iterdir()):
            if not trial_dir.is_dir():
                continue

            game_file = trial_dir / 'game.tw-pddl'
            traj_data_file = trial_dir / 'traj_data.json'
            if not game_file.exists() or not traj_data_file.exists():
                skipped += 1
                continue

            try:
                with open(traj_data_file, 'r') as f:
                    traj_data = json.load(f)
                anns = traj_data['turk_annotations']['anns']
                ann_task_desc = anns[0]['task_desc'].strip()

                main_env.game_files = [str(game_file)]
                env = main_env.init_env(batch_size=1)
                raw_obs = _extract_init_obs(env.reset())
                room_desc, obs_task_desc = _split_room_and_task(raw_obs)

                # 与 alfworld_tasks_suffix.json 保持一致：优先使用环境自带任务描述
                final_task_desc = obs_task_desc if obs_task_desc else ann_task_desc

                if not room_desc or not final_task_desc:
                    skipped += 1
                    continue

                goal = f"{room_desc}\n\nYour task is to: {final_task_desc}.___{task_id}"
                print(f"goal : {goal}")
                tasks.append({
                    'goal': goal,
                    'gamefile': str(game_file.absolute())
                })
                task_id += 1

                if task_id % 200 == 0:
                    print(f"已生成 {task_id} 条任务...")
            except Exception as e:
                skipped += 1
                print(f"跳过 {trial_dir.name}: {e}")

    print(f"完成。有效任务: {len(tasks)}，跳过: {skipped}")
    return tasks


def main():
    print('开始收集 train 数据集任务...')
    print(f'数据路径: {TRAIN_DATA_PATH}')

    tasks = collect_train_tasks(TRAIN_DATA_PATH)
    print(f"\n总共收集到 {len(tasks)} 个任务")

    if not tasks:
        print('警告：没有收集到任何任务！')
        return

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(tasks, f, indent=2)

    print(f"任务列表已保存到: {OUTPUT_JSON}")
    print('\n前 3 个任务示例:')
    for i, task in enumerate(tasks[:3]):
        print(f"\n任务 {i}:")
        print(f"  gamefile: {task['gamefile']}")
        print(f"  goal (前150字符): {task['goal'][:150]}...")


if __name__ == '__main__':
    main()
