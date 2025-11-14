"""
情感数据集DataLoader使用示例
展示如何使用emotion_dataloader模块加载MOSEI和MELD数据

数据加载策略：
- MOSEI: 根据指定label加载数据，按7/3划分训练集和测试集
- MELD: train+dev合并为训练集，test为测试集
"""

import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_dataloader import (
    create_dataloaders,
    create_multiple_dataloaders,
    custom_collate_fn
)


def example1_mosei_dataloader():
    """示例1: 创建MOSEI数据的DataLoader"""
    print("=" * 60)
    print("示例1: 创建MOSEI数据的DataLoader (7/3划分)")
    print("=" * 60)

    # 创建happy情感，label_id为0的MOSEI数据加载器
    # 自动按7/3划分为训练集和测试集
    dataloaders = create_dataloaders(
        data_dir='Data',
        dataset_name='MOSEI',
        emotion='happy',
        label_id=0,
        batch_size=16,
        num_workers=2,
        train_ratio=0.7,  # 70%训练，30%测试
        seed=42
    )

    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    print(f"训练集DataLoader创建成功!")
    print(f"  - 训练集大小: {len(train_loader.dataset)}")
    print(f"  - 训练批次数: {len(train_loader)}")

    print(f"\n测试集DataLoader创建成功!")
    print(f"  - 测试集大小: {len(test_loader.dataset)}")
    print(f"  - 测试批次数: {len(test_loader)}")

    # 获取一个训练batch
    for batch in train_loader:
        print(f"\n第一个训练batch的keys: {batch.keys() if isinstance(batch, dict) else 'N/A'}")
        if isinstance(batch, dict):
            for key, value in batch.items():
                print(f"  {key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
        break

    print("\n" + "=" * 60 + "\n")


def example2_meld_dataloader():
    """示例2: 创建MELD数据的DataLoader"""
    print("=" * 60)
    print("示例2: 创建MELD数据的DataLoader (train+dev合并)")
    print("=" * 60)

    # 创建sad情感，label_id为1的MELD数据加载器
    # train+dev自动合并为训练集，test为测试集
    dataloaders = create_dataloaders(
        data_dir='Data',
        dataset_name='MELD',
        emotion='sad',
        label_id=1,
        batch_size=32,
        num_workers=4
    )

    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    print(f"训练集DataLoader创建成功!")
    print(f"  - 训练集大小 (train+dev): {len(train_loader.dataset)}")
    print(f"  - 训练批次数: {len(train_loader)}")

    print(f"\n测试集DataLoader创建成功!")
    print(f"  - 测试集大小: {len(test_loader.dataset)}")
    print(f"  - 测试批次数: {len(test_loader)}")

    # 打印数据集信息
    dataset_info = train_loader.dataset.get_info()
    print(f"\n数据集信息:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60 + "\n")


def example3_multiple_mosei_dataloaders():
    """示例3: 批量创建多个MOSEI DataLoader"""
    print("=" * 60)
    print("示例3: 批量创建多个MOSEI DataLoader")
    print("=" * 60)

    # 定义需要加载的情感和标签对
    emotion_label_pairs = [
        ('happy', 0),
        ('sad', 1),
        ('anger', 2),
        ('disgust', 3),
        ('surprise', 4),
        ('fear', 5)
    ]

    # 批量创建DataLoader
    all_dataloaders = create_multiple_dataloaders(
        data_dir='Data',
        dataset_name='MOSEI',
        emotion_label_pairs=emotion_label_pairs,
        batch_size=16,
        num_workers=2,
        train_ratio=0.7,
        seed=42
    )

    print(f"成功创建 {len(all_dataloaders)} 组DataLoader:")
    for key, loaders in all_dataloaders.items():
        train_size = len(loaders['train'].dataset)
        test_size = len(loaders['test'].dataset)
        total_size = train_size + test_size
        print(f"  {key}:")
        print(f"    - 训练集: {train_size} 样本 ({train_size/total_size*100:.1f}%)")
        print(f"    - 测试集: {test_size} 样本 ({test_size/total_size*100:.1f}%)")

    # 访问特定的DataLoader
    print("\n使用happy_0的训练集:")
    happy_train_loader = all_dataloaders['happy_0']['train']
    print(f"  批次数: {len(happy_train_loader)}")

    print("\n" + "=" * 60 + "\n")


def example4_multiple_meld_dataloaders():
    """示例4: 批量创建多个MELD DataLoader"""
    print("=" * 60)
    print("示例4: 批量创建多个MELD DataLoader")
    print("=" * 60)

    # 定义需要加载的情感和标签对
    emotion_label_pairs = [
        ('happy', 0),
        ('sad', 1),
        ('anger', 2)
    ]

    # 批量创建DataLoader
    all_dataloaders = create_multiple_dataloaders(
        data_dir='Data',
        dataset_name='MELD',
        emotion_label_pairs=emotion_label_pairs,
        batch_size=32,
        num_workers=4
    )

    print(f"成功创建 {len(all_dataloaders)} 组DataLoader:")
    for key, loaders in all_dataloaders.items():
        train_size = len(loaders['train'].dataset)
        test_size = len(loaders['test'].dataset)
        print(f"  {key}:")
        print(f"    - 训练集 (train+dev): {train_size} 样本")
        print(f"    - 测试集: {test_size} 样本")

    print("\n" + "=" * 60 + "\n")


def example5_custom_collate():
    """示例5: 使用自定义collate函数"""
    print("=" * 60)
    print("示例5: 使用自定义collate函数")
    print("=" * 60)

    dataloaders = create_dataloaders(
        data_dir='Data',
        dataset_name='MOSEI',
        emotion='happy',
        label_id=0,
        batch_size=8,
        num_workers=2,
        collate_fn=custom_collate_fn  # 使用自定义collate函数
    )

    train_loader = dataloaders['train']

    print(f"DataLoader创建成功!")
    print("使用了自定义collate函数来整理batch数据")

    # 获取一个batch
    for batch in train_loader:
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Audio shape: {batch['audio'].shape}")
        print(f"Text shape: {batch['text'].shape}")
        print(f"Video shape: {batch['video'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        break

    print("\n" + "=" * 60 + "\n")


def example6_training_loop():
    """示例6: 在训练循环中使用DataLoader"""
    print("=" * 60)
    print("示例6: 在训练循环中使用DataLoader")
    print("=" * 60)

    # 创建MOSEI训练和测试DataLoader
    mosei_loaders = create_dataloaders(
        data_dir='Data',
        dataset_name='MOSEI',
        emotion='happy',
        label_id=0,
        batch_size=32,
        num_workers=4,
        train_ratio=0.7
    )

    train_loader = mosei_loaders['train']
    test_loader = mosei_loaders['test']

    # 模拟训练循环
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练阶段
        print("  Training...")
        for batch_idx, batch in enumerate(train_loader):
            # 这里进行模型训练
            # loss = model(batch)
            # optimizer.step()

            if batch_idx == 0:  # 只打印第一个batch作为示例
                print(f"    Batch 1 loaded, type: {type(batch)}")

        # 测试阶段
        print("  Testing...")
        for batch_idx, batch in enumerate(test_loader):
            # 这里进行测试
            # test_loss = model(batch)

            if batch_idx == 0:
                print(f"    Batch 1 loaded, type: {type(batch)}")

    print("\n" + "=" * 60 + "\n")


def example7_with_transform():
    """示例7: 使用数据转换函数"""
    print("=" * 60)
    print("示例7: 使用数据转换函数")
    print("=" * 60)

    # 定义一个简单的transform函数
    def my_transform(sample):
        """
        对样本进行转换
        例如：数据增强、归一化等
        """
        if isinstance(sample, dict):
            # 假设对音频特征进行归一化
            if 'audio_features' in sample:
                sample['audio_features'] = sample['audio_features'] / sample['audio_features'].max()

            # 可以添加更多转换
            # sample['augmented'] = True

        return sample

    dataloaders = create_dataloaders(
        data_dir='Data',
        dataset_name='MOSEI',
        emotion='surprise',
        label_id=4,
        batch_size=16,
        transform=my_transform  # 应用转换函数
    )

    train_loader = dataloaders['train']

    print(f"DataLoader创建成功，已应用自定义transform函数")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(dataloaders['test'].dataset)}")

    print("\n" + "=" * 60 + "\n")


def example8_compare_datasets():
    """示例8: 比较MOSEI和MELD的数据划分"""
    print("=" * 60)
    print("示例8: 比较MOSEI和MELD的数据划分策略")
    print("=" * 60)

    # MOSEI数据集
    print("\nMOSEI数据集 (7/3划分):")
    mosei_loaders = create_dataloaders(
        data_dir='Data',
        dataset_name='MOSEI',
        emotion='happy',
        label_id=0,
        batch_size=32,
        train_ratio=0.7
    )
    mosei_train_size = len(mosei_loaders['train'].dataset)
    mosei_test_size = len(mosei_loaders['test'].dataset)
    print(f"  训练集: {mosei_train_size} 样本")
    print(f"  测试集: {mosei_test_size} 样本")
    print(f"  比例: {mosei_train_size/(mosei_train_size+mosei_test_size):.1%} / {mosei_test_size/(mosei_train_size+mosei_test_size):.1%}")

    # MELD数据集
    print("\nMELD数据集 (train+dev / test):")
    meld_loaders = create_dataloaders(
        data_dir='Data',
        dataset_name='MELD',
        emotion='happy',
        label_id=0,
        batch_size=32
    )
    meld_train_size = len(meld_loaders['train'].dataset)
    meld_test_size = len(meld_loaders['test'].dataset)
    print(f"  训练集 (train+dev): {meld_train_size} 样本")
    print(f"  测试集: {meld_test_size} 样本")
    print(f"  比例: {meld_train_size/(meld_train_size+meld_test_size):.1%} / {meld_test_size/(meld_train_size+meld_test_size):.1%}")

    print("\n" + "=" * 60 + "\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("情感数据集DataLoader使用示例")
    print("=" * 60 + "\n")

    # 运行所有示例（注意：需要实际的pkl文件才能运行）
    print("注意：以下示例需要实际的pkl文件才能正常运行")
    print("请确保Data目录下有对应格式的pkl文件\n")

    try:
        example1_mosei_dataloader()
    except Exception as e:
        print(f"示例1出错: {e}\n")

    try:
        example2_meld_dataloader()
    except Exception as e:
        print(f"示例2出错: {e}\n")

    try:
        example3_multiple_mosei_dataloaders()
    except Exception as e:
        print(f"示例3出错: {e}\n")

    try:
        example4_multiple_meld_dataloaders()
    except Exception as e:
        print(f"示例4出错: {e}\n")

    try:
        example5_custom_collate()
    except Exception as e:
        print(f"示例5出错: {e}\n")

    try:
        example6_training_loop()
    except Exception as e:
        print(f"示例6出错: {e}\n")

    try:
        example7_with_transform()
    except Exception as e:
        print(f"示例7出错: {e}\n")

    try:
        example8_compare_datasets()
    except Exception as e:
        print(f"示例8出错: {e}\n")

    print("\n所有示例运行完毕!")
