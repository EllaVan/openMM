"""
示例脚本：创建和使用任务配置文件

这个脚本展示了如何：
1. 创建任务配置
2. 保存为JSON文件
3. 加载JSON文件
4. 在训练中使用
"""

import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from continual_learning.domain_splitter import (
    DomainSplitter,
    TaskConfig,
    create_predefined_task_sequence
)
from fusion.dataloader import load_mosei_data


def example1_create_from_strategy():
    """示例1：使用策略自动生成配置"""
    print("="*80)
    print("示例1：使用策略自动生成任务配置")
    print("="*80)

    # 加载数据集
    print("\n1. 加载数据集...")
    dataset = load_mosei_data(
        data_dir='../../output/mosei_features',
        emotion='all'
    )

    # 创建splitter
    print("\n2. 创建DomainSplitter...")
    splitter = DomainSplitter(dataset, exclude_neutral=True)

    # 使用策略生成任务
    print("\n3. 使用 'small_unseen' 策略生成任务...")
    tasks = splitter.create_tasks_by_strategy(
        strategy='small_unseen',
        num_tasks=3,
        seen_classes_base=[0, 1]  # happy, sad
    )

    # 查看生成的任务
    print("\n4. 生成的任务配置:")
    for task in tasks:
        print(f"  {task}")

    # 保存配置
    config_path = 'strategy_task_config.json'
    print(f"\n5. 保存配置到 {config_path}...")
    splitter.save_task_configs(tasks, config_path)

    print("\n✓ 完成！配置文件已保存。")
    return config_path


def example2_create_predefined():
    """示例2：使用预定义序列"""
    print("\n" + "="*80)
    print("示例2：使用预定义任务序列")
    print("="*80)

    # 使用预定义序列
    print("\n可用的预定义序列: 'demo', 'full', 'custom'")

    for seq_name in ['demo', 'full', 'custom']:
        print(f"\n序列 '{seq_name}':")
        tasks = create_predefined_task_sequence(seq_name, 'MOSEI')
        for task in tasks:
            print(f"  {task}")

    # 保存custom序列
    tasks = create_predefined_task_sequence('custom', 'MOSEI')
    config_path = 'predefined_task_config.json'

    print(f"\n保存 'custom' 序列到 {config_path}...")

    # 需要创建一个临时splitter来保存（仅用于格式）
    # 或者直接手动创建JSON
    import json
    output = {
        'num_tasks': len(tasks),
        'exclude_neutral': True,
        'random_seed': 42,
        'tasks': [task.to_dict() for task in tasks]
    }

    with open(config_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("✓ 完成！")
    return config_path


def example3_create_manual():
    """示例3：手动创建任务配置"""
    print("\n" + "="*80)
    print("示例3：手动创建任务配置")
    print("="*80)

    # 手动创建任务
    print("\n手动创建3个任务...")

    tasks = [
        TaskConfig(
            task_id=0,
            task_name="CustomTask0_Happy_Sad_vs_Angry",
            dataset_name="MOSEI",
            seen_classes=[0, 1],      # happy=0, sad=1
            unseen_classes=[2],        # angry=2
            data_split=None
        ),
        TaskConfig(
            task_id=1,
            task_name="CustomTask1_Happy_vs_Disgust_Fear",
            dataset_name="MOSEI",
            seen_classes=[0],          # happy=0
            unseen_classes=[4, 5],     # disgust=4, fear=5
            data_split=None
        ),
        TaskConfig(
            task_id=2,
            task_name="CustomTask2_Happy_Sad_vs_Surprise",
            dataset_name="MOSEI",
            seen_classes=[0, 1],      # happy=0, sad=1
            unseen_classes=[3],        # surprise=3
            data_split=None
        )
    ]

    # 查看任务
    print("\n创建的任务:")
    for task in tasks:
        print(f"  {task}")

    # 保存
    config_path = 'manual_task_config.json'
    print(f"\n保存到 {config_path}...")

    import json
    output = {
        'num_tasks': len(tasks),
        'exclude_neutral': True,
        'random_seed': 42,
        'tasks': [task.to_dict() for task in tasks]
    }

    with open(config_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("✓ 完成！")
    return config_path


def example4_load_and_use(config_path):
    """示例4：加载配置并使用"""
    print("\n" + "="*80)
    print("示例4：加载配置文件并使用")
    print("="*80)

    # 加载配置
    print(f"\n1. 从 {config_path} 加载配置...")
    tasks = DomainSplitter.load_task_configs(config_path)

    print(f"\n2. 加载了 {len(tasks)} 个任务:")
    for task in tasks:
        print(f"  {task}")

    # 使用配置创建dataloader
    print("\n3. 使用配置创建DataLoader...")
    print("   (需要实际数据集，此处仅演示)")

    # dataset = load_mosei_data(...)
    # splitter = DomainSplitter(dataset, exclude_neutral=True)
    #
    # for task_config in tasks:
    #     seen_loader, unseen_loader = splitter.create_task_dataloaders(
    #         task_config,
    #         batch_size=32
    #     )
    #     # 训练...

    print("✓ 完成！")


def example5_modify_json():
    """示例5：手动修改JSON文件"""
    print("\n" + "="*80)
    print("示例5：修改JSON文件的字段说明")
    print("="*80)

    print("\nJSON文件字段说明:")
    print("-" * 80)

    field_explanations = {
        "task_id": "任务ID (0, 1, 2, ...)",
        "task_name": "任务名称 (自定义字符串)",
        "dataset_name": "数据集名称 ('MOSEI' 或 'MELD')",
        "seen_classes": "有标签的情绪类别ID列表 [0=happy, 1=sad, 2=angry, 3=surprise, 4=disgust, 5=fear]",
        "unseen_classes": "无标签的情绪类别ID列表",
        "data_split": "数据划分 (null=使用全部, 0.5=使用50%, 等)"
    }

    for field, explanation in field_explanations.items():
        print(f"  {field:20} : {explanation}")

    print("\n" + "-" * 80)
    print("情绪类别ID映射:")
    print("-" * 80)
    print("  0: happy")
    print("  1: sad")
    print("  2: angry")
    print("  3: surprise")
    print("  4: disgust")
    print("  5: fear")
    print("  6: neutral (通常被排除)")

    print("\n" + "-" * 80)
    print("修改建议:")
    print("-" * 80)
    print("  1. 修改 seen_classes: 改变有标签的情绪")
    print("  2. 修改 unseen_classes: 改变零样本学习的情绪")
    print("  3. 修改 task_name: 使用更有意义的名称")
    print("  4. 修改 data_split: 控制每个任务使用的数据量")
    print("\n✓ 完成！")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("任务配置文件创建与使用完整示例")
    print("="*80)

    # 示例1：策略生成
    config1 = example1_create_from_strategy()

    # 示例2：预定义序列
    config2 = example2_create_predefined()

    # 示例3：手动创建
    config3 = example3_create_manual()

    # 示例4：加载使用
    example4_load_and_use(config3)

    # 示例5：修改说明
    example5_modify_json()

    print("\n" + "="*80)
    print("所有示例完成！")
    print("="*80)
    print("\n生成的配置文件:")
    print(f"  1. {config1}")
    print(f"  2. {config2}")
    print(f"  3. {config3}")
    print("\n使用方法:")
    print("  python blackbox_main.py --task_config_path <配置文件路径>")
    print("="*80)


if __name__ == "__main__":
    main()
