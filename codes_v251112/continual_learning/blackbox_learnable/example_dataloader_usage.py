"""
持续学习数据加载器使用示例

演示如何使用 dataloader_continual.py 加载多任务数据
"""

import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

from dataloader_continual import (
    create_task_dataloaders,
    load_all_tasks,
    IncrementalLabelMapper
)


def example1_single_task():
    """示例1: 加载单个任务"""
    print("="*80)
    print("示例1: 加载单个任务 (Task 0)")
    print("="*80)

    task_config_path = "../../../codes_v251119/config/task_config.json"

    # 加载Task 0
    train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
        task_config_path=task_config_path,
        task_id=0,
        batch_size=8,
        num_workers=0,  # 设为0便于调试
        train_ratio=0.8
    )

    print(f"\n任务信息:")
    print(f"  任务名称: {task_info['task_name']}")
    print(f"  Seen情绪: {task_info['seen_emotions']}")
    print(f"  Unseen情绪: {task_info['unseen_emotions']}")
    print(f"  训练样本数: {task_info['train_stats']['total']}")
    print(f"  测试样本数: {task_info['test_stats']['total']}")

    # 查看一个batch
    print(f"\n查看训练集第一个batch:")
    batch = next(iter(train_loader))
    print(f"  text shape: {batch['text'].shape}")
    print(f"  audio shape: {batch['audio'].shape}")
    print(f"  video shape: {batch['video'].shape}")
    print(f"  labels: {batch['label'].tolist()}")
    print(f"  original_labels: {batch['original_label'].tolist()}")
    print(f"  is_seen: {batch['is_seen'].tolist()}")

    # 查看标签映射
    print(f"\n标签映射:")
    print(f"  全局映射: {label_mapper.original_to_incremental}")
    print(f"  总类数: {label_mapper.get_num_classes_so_far()}")

    print("\n✓ 示例1完成!\n")
    return label_mapper


def example2_sequential_tasks():
    """示例2: 顺序加载多个任务（模拟持续学习）"""
    print("="*80)
    print("示例2: 顺序加载多个任务")
    print("="*80)

    task_config_path = "../../../codes_v251119/config/task_config.json"

    # 创建全局标签映射器
    label_mapper = IncrementalLabelMapper()

    # 加载Task 0
    print("\n[加载 Task 0]")
    train_loader_0, test_loader_0, label_mapper, info_0 = create_task_dataloaders(
        task_config_path=task_config_path,
        task_id=0,
        label_mapper=label_mapper,
        batch_size=8,
        num_workers=0
    )

    print(f"\nTask 0 完成:")
    print(f"  总类数: {label_mapper.get_num_classes_so_far()}")
    print(f"  全局映射: {label_mapper.original_to_incremental}")

    # 加载Task 1 (使用同一个label_mapper)
    print("\n[加载 Task 1]")
    train_loader_1, test_loader_1, label_mapper, info_1 = create_task_dataloaders(
        task_config_path=task_config_path,
        task_id=1,
        label_mapper=label_mapper,  # 关键: 传入之前的mapper
        batch_size=8,
        num_workers=0
    )

    print(f"\nTask 1 完成:")
    print(f"  总类数: {label_mapper.get_num_classes_so_far()}")
    print(f"  全局映射: {label_mapper.original_to_incremental}")

    # 观察标签一致性
    print(f"\n观察标签一致性:")
    print(f"  Task 0 seen: {info_0['mapping_info']['seen_mapping']}")
    print(f"  Task 1 seen: {info_1['mapping_info']['seen_mapping']}")
    print(f"  注意: happy 在两个任务中都是标签 0 (保持一致)")

    print("\n✓ 示例2完成!\n")


def example3_load_all():
    """示例3: 一次性加载所有任务"""
    print("="*80)
    print("示例3: 一次性加载所有任务")
    print("="*80)

    task_config_path = "../../../codes_v251119/config/task_config.json"

    # 加载所有任务
    all_tasks = load_all_tasks(
        task_config_path=task_config_path,
        batch_size=8,
        num_workers=0,
        train_ratio=0.8
    )

    print(f"\n加载了 {len(all_tasks)} 个任务:")
    for i, (train_loader, test_loader, task_info) in enumerate(all_tasks):
        print(f"\nTask {i}:")
        print(f"  名称: {task_info['task_name']}")
        print(f"  训练批次: {len(train_loader)}")
        print(f"  测试批次: {len(test_loader)}")
        print(f"  Seen情绪: {task_info['seen_emotions']}")
        print(f"  Unseen情绪: {task_info['unseen_emotions']}")
        print(f"  当前总类数: {task_info['num_classes_so_far']}")

    print("\n✓ 示例3完成!\n")


def example4_seen_unseen_separation():
    """示例4: Seen和Unseen样本分离"""
    print("="*80)
    print("示例4: Seen和Unseen样本分离")
    print("="*80)

    task_config_path = "../../../codes_v251119/config/task_config.json"

    # 加载Task 0
    train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
        task_config_path=task_config_path,
        task_id=0,
        batch_size=16,
        num_workers=0
    )

    # 获取一个batch
    batch = next(iter(train_loader))

    # 分离seen和unseen
    is_seen = batch['is_seen']
    seen_mask = is_seen
    unseen_mask = ~is_seen

    print(f"\nBatch统计:")
    print(f"  总样本数: {len(batch['label'])}")
    print(f"  Seen样本数: {seen_mask.sum().item()}")
    print(f"  Unseen样本数: {unseen_mask.sum().item()}")

    if seen_mask.any():
        print(f"\nSeen样本:")
        print(f"  标签: {batch['label'][seen_mask].tolist()}")
        print(f"  原始标签: {batch['original_label'][seen_mask].tolist()}")

    if unseen_mask.any():
        print(f"\nUnseen样本:")
        print(f"  标签: {batch['label'][unseen_mask].tolist()}")
        print(f"  原始标签: {batch['original_label'][unseen_mask].tolist()}")

    print(f"\n在训练中的使用:")
    print(f"  Seen样本: 使用真实标签，高权重训练 (weight=1.0)")
    print(f"  Unseen样本: 使用伪标签，低权重训练 (weight=0.3)")

    print("\n✓ 示例4完成!\n")


def example5_label_mapping_details():
    """示例5: 标签映射细节"""
    print("="*80)
    print("示例5: 标签映射细节")
    print("="*80)

    from dataloader_continual import IncrementalLabelMapper

    # 手动演示标签映射过程
    mapper = IncrementalLabelMapper()

    # Task 0
    print("\nTask 0:")
    print("  Seen: happy(0), sad(1)")
    print("  Unseen: surprise(3), disgust(4)")

    mapping_0 = mapper.add_task(
        task_id=0,
        seen_emotions={'happy': 0, 'sad': 1},
        unseen_emotions={'surprise': 3, 'disgust': 4}
    )

    # Task 1
    print("\nTask 1:")
    print("  Seen: happy(0), anger(2)")
    print("  Unseen: fear(5)")

    mapping_1 = mapper.add_task(
        task_id=1,
        seen_emotions={'happy': 0, 'anger': 2},
        unseen_emotions={'fear': 5}
    )

    # 最终映射
    print("\n最终全局映射:")
    for original, incremental in sorted(mapper.original_to_incremental.items()):
        emotion_name = {0: 'happy', 1: 'sad', 2: 'anger', 3: 'surprise', 4: 'disgust', 5: 'fear'}[original]
        is_seen_str = "✓ seen" if mapper.is_seen(original) else "✗ unseen only"
        print(f"  {emotion_name}(原始={original}) -> 增量标签={incremental} [{is_seen_str}]")

    print(f"\n总类数: {mapper.get_num_classes_so_far()}")

    print("\n✓ 示例5完成!\n")


def main():
    """运行所有示例"""
    print("\n" + "#"*80)
    print("# 持续学习数据加载器使用示例")
    print("#"*80 + "\n")

    try:
        # 示例1: 单任务
        example1_single_task()

        # 示例2: 顺序多任务
        example2_sequential_tasks()

        # 示例3: 一次性加载所有
        example3_load_all()

        # 示例4: Seen/Unseen分离
        example4_seen_unseen_separation()

        # 示例5: 标签映射细节
        example5_label_mapping_details()

        print("\n" + "#"*80)
        print("# 所有示例运行完成!")
        print("#"*80)

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n请确保:")
        print("  1. 数据文件存在于配置的data_dir路径")
        print("  2. task_config.json路径正确")
        print("  3. 文件命名格式正确: {DATASET}{emotion}label{id}.pkl")

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
