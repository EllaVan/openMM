"""
跨数据集持续学习示例

演示如何在不同数据集之间进行持续学习：
- Task 0: MOSEI
- Task 1: MELD
- Task 2: MOSEI
- Task 3: MELD

关键点：
1. 每个任务可以使用不同的数据集
2. 标签映射在所有数据集间保持一致
3. 测试跨数据集的知识迁移
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


def example_cross_dataset_sequential():
    """示例: 跨数据集顺序加载"""
    print("="*80)
    print("跨数据集持续学习示例")
    print("="*80)

    task_config_path = "./task_config_cross_dataset.json"

    # 创建全局标签映射器
    label_mapper = IncrementalLabelMapper()

    # 存储所有任务的信息
    all_task_info = []

    # 顺序加载4个任务
    for task_id in range(4):
        print(f"\n{'='*80}")
        print(f"加载 Task {task_id}")
        print(f"{'='*80}")

        try:
            train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
                task_config_path=task_config_path,
                task_id=task_id,
                label_mapper=label_mapper,
                batch_size=8,
                num_workers=0,
                train_ratio=0.8
            )

            all_task_info.append(task_info)

            # 显示任务详情
            print(f"\n任务详情:")
            print(f"  数据集: {task_info['dataset_name']}")
            print(f"  数据目录: {task_info['data_dir']}")
            print(f"  Seen情绪: {task_info['seen_emotions']}")
            print(f"  Unseen情绪: {task_info['unseen_emotions']}")
            print(f"  训练样本: {task_info['train_stats']['total']}")
            print(f"  测试样本: {task_info['test_stats']['total']}")
            print(f"  当前总类数: {task_info['num_classes_so_far']}")

            # 查看一个batch
            if len(train_loader) > 0:
                batch = next(iter(train_loader))
                print(f"\n  Batch示例:")
                print(f"    Shape: text={batch['text'].shape}, audio={batch['audio'].shape}, video={batch['video'].shape}")
                print(f"    Labels: {batch['label'].tolist()[:5]}... (前5个)")
                print(f"    Is seen: {batch['is_seen'].tolist()[:5]}... (前5个)")

        except FileNotFoundError as e:
            print(f"\n  ⚠️  警告: 数据文件未找到")
            print(f"     {e}")
            print(f"     跳过此任务，继续下一个...")
            continue

    # 总结
    print(f"\n{'='*80}")
    print(f"跨数据集持续学习总结")
    print(f"{'='*80}")

    print(f"\n全局标签映射:")
    for original, incremental in sorted(label_mapper.original_to_incremental.items()):
        emotion_names = {0: 'happy', 1: 'sad', 2: 'anger', 3: 'surprise', 4: 'disgust', 5: 'fear'}
        emotion_name = emotion_names.get(original, f'unknown_{original}')
        is_seen = "✓ seen" if label_mapper.is_seen(original) else "✗ unseen only"
        print(f"  {emotion_name}(原始={original}) -> 增量标签={incremental} [{is_seen}]")

    print(f"\n总类数: {label_mapper.get_num_classes_so_far()}")

    print(f"\n任务数据集分布:")
    for i, info in enumerate(all_task_info):
        print(f"  Task {i}: {info['dataset_name']:6} - {info['task_name']}")

    # 观察数据集切换
    print(f"\n数据集切换观察:")
    for i in range(len(all_task_info) - 1):
        curr_dataset = all_task_info[i]['dataset_name']
        next_dataset = all_task_info[i + 1]['dataset_name']
        if curr_dataset != next_dataset:
            print(f"  Task {i} -> Task {i+1}: {curr_dataset} -> {next_dataset} (数据集切换)")
        else:
            print(f"  Task {i} -> Task {i+1}: {curr_dataset} (同数据集)")

    print("\n✓ 跨数据集持续学习示例完成!\n")


def example_cross_dataset_analysis():
    """示例: 跨数据集分析"""
    print("="*80)
    print("跨数据集配置分析")
    print("="*80)

    import json

    task_config_path = "./task_config_cross_dataset.json"

    # 读取配置
    with open(task_config_path, 'r') as f:
        config = json.load(f)

    print(f"\n配置概览:")
    print(f"  总任务数: {config['num_tasks']}")
    print(f"  默认数据集: {config['default_dataset']}")
    print(f"  默认数据目录: {config['default_data_dir']}")

    print(f"\n任务配置:")
    for task in config['tasks']:
        print(f"\n  Task {task['task_id']}: {task['task_name']}")
        print(f"    数据集: {task['dataset_name']}")
        print(f"    数据目录: {task['data_dir']}")
        print(f"    Seen: {list(task['seen_emotions'].keys())}")
        print(f"    Unseen: {list(task['unseen_emotions'].keys())}")

    # 分析数据集使用
    dataset_usage = {}
    for task in config['tasks']:
        dataset = task['dataset_name']
        dataset_usage[dataset] = dataset_usage.get(dataset, 0) + 1

    print(f"\n数据集使用统计:")
    for dataset, count in dataset_usage.items():
        print(f"  {dataset}: {count} 个任务")

    print("\n✓ 跨数据集配置分析完成!\n")


def example_dataset_specific_handling():
    """示例: 数据集特定处理"""
    print("="*80)
    print("数据集特定处理示例")
    print("="*80)

    task_config_path = "./task_config_cross_dataset.json"

    label_mapper = IncrementalLabelMapper()

    for task_id in [0, 1]:  # MOSEI和MELD
        try:
            train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
                task_config_path=task_config_path,
                task_id=task_id,
                label_mapper=label_mapper,
                batch_size=4,
                num_workers=0
            )

            dataset_name = task_info['dataset_name']

            print(f"\nTask {task_id} - {dataset_name}:")

            # 获取一个batch
            if len(train_loader) > 0:
                batch = next(iter(train_loader))

                # 根据数据集类型进行特定处理
                if dataset_name == 'MOSEI':
                    print(f"  处理MOSEI数据...")
                    print(f"    MOSEI特点: 连续值情绪标注，样本级别特征")
                    print(f"    特征维度: text={batch['text'].shape[1]}, audio={batch['audio'].shape[1]}, video={batch['video'].shape[1]}")

                elif dataset_name == 'MELD':
                    print(f"  处理MELD数据...")
                    print(f"    MELD特点: 对话情绪识别，话语级别特征")
                    print(f"    特征维度: text={batch['text'].shape[1]}, audio={batch['audio'].shape[1]}, video={batch['video'].shape[1]}")

                # 通用处理
                print(f"    Batch大小: {len(batch['label'])}")
                print(f"    Seen样本数: {batch['is_seen'].sum().item()}")
                print(f"    Unseen样本数: {(~batch['is_seen']).sum().item()}")

        except FileNotFoundError as e:
            print(f"\n  ⚠️  数据文件未找到，跳过Task {task_id}")
            continue

    print("\n✓ 数据集特定处理示例完成!\n")


def example_training_loop():
    """示例: 跨数据集训练循环（伪代码）"""
    print("="*80)
    print("跨数据集训练循环示例（伪代码）")
    print("="*80)

    print("""
完整的跨数据集训练流程：

```python
from dataloader_continual import IncrementalLabelMapper, create_task_dataloaders
import torch
import torch.nn as nn

# 初始化
label_mapper = IncrementalLabelMapper()
model = YourMultimodalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 持续学习循环
for task_id in range(4):  # 4个任务: MOSEI, MELD, MOSEI, MELD
    print(f"\\n训练 Task {task_id}")

    # 加载任务数据
    train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
        task_config_path='task_config_cross_dataset.json',
        task_id=task_id,
        label_mapper=label_mapper,
        batch_size=32
    )

    dataset_name = task_info['dataset_name']
    print(f"  数据集: {dataset_name}")

    # 根据数据集调整训练策略
    if dataset_name == 'MOSEI':
        lr = 1e-4
        epochs = 10
    elif dataset_name == 'MELD':
        lr = 5e-5  # MELD可能需要更小的学习率
        epochs = 15

    # 更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 训练
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            text = batch['text'].cuda()
            audio = batch['audio'].cuda()
            video = batch['video'].cuda()
            labels = batch['label'].cuda()
            is_seen = batch['is_seen'].cuda()

            # 前向传播
            outputs = model(text, audio, video)

            # 损失（seen和unseen不同权重）
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
            weights = torch.where(is_seen, 1.0, 0.3)
            loss = (loss * weights).mean()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 评估
    model.eval()
    # ... 评估代码

    # 保存检查点
    torch.save({
        'task_id': task_id,
        'dataset': dataset_name,
        'model': model.state_dict(),
        'label_mapper': label_mapper.original_to_incremental
    }, f'checkpoint_task{task_id}.pt')

print("\\n跨数据集持续学习完成!")
```

关键点：
1. 使用同一个label_mapper确保标签一致性
2. 根据数据集调整超参数（学习率、epochs等）
3. Seen/Unseen样本使用不同权重
4. 保存每个任务的检查点用于评估遗忘
""")

    print("\n✓ 训练循环示例完成!\n")


def main():
    """运行所有示例"""
    print("\n" + "#"*80)
    print("# 跨数据集持续学习完整示例")
    print("#"*80 + "\n")

    # 示例1: 配置分析
    example_cross_dataset_analysis()

    # 示例2: 顺序加载
    print("\n" + "="*80 + "\n")
    example_cross_dataset_sequential()

    # 示例3: 数据集特定处理
    print("\n" + "="*80 + "\n")
    example_dataset_specific_handling()

    # 示例4: 训练循环
    print("\n" + "="*80 + "\n")
    example_training_loop()

    print("\n" + "#"*80)
    print("# 所有示例运行完成!")
    print("# ")
    print("# 要点总结:")
    print("# 1. 每个任务可以使用不同的数据集（MOSEI、MELD等）")
    print("# 2. 标签映射在所有数据集间保持一致")
    print("# 3. 根据数据集特性调整训练策略")
    print("# 4. 测试跨数据集的知识迁移和域适应")
    print("#"*80)


if __name__ == "__main__":
    main()
