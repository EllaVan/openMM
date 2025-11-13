"""
情感数据集DataLoader使用示例
展示如何使用emotion_dataloader模块加载MOSEI和MELD数据
"""

import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_dataloader import (
    create_emotion_dataloader,
    create_multiple_dataloaders,
    create_all_splits_dataloaders,
    custom_collate_fn
)


def example1_single_mosei_dataloader():
    """示例1: 创建单个MOSEI数据的DataLoader"""
    print("=" * 60)
    print("示例1: 创建单个MOSEI数据的DataLoader")
    print("=" * 60)

    # 创建happy情感，label_id为0的MOSEI数据加载器
    dataloader = create_emotion_dataloader(
        data_dir='Data',
        dataset_name='MOSEI',
        emotion='happy',
        label_id=0,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )

    print(f"DataLoader创建成功!")
    print(f"数据集大小: {len(dataloader.dataset)}")
    print(f"批次数量: {len(dataloader)}")

    # 获取一个batch
    for batch in dataloader:
        print(f"\n第一个batch的keys: {batch.keys() if isinstance(batch, dict) else 'N/A'}")
        if isinstance(batch, dict):
            for key, value in batch.items():
                print(f"  {key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
        break

    print("\n" + "=" * 60 + "\n")


def example2_single_meld_dataloader():
    """示例2: 创建单个MELD数据的DataLoader"""
    print("=" * 60)
    print("示例2: 创建单个MELD数据的DataLoader")
    print("=" * 60)

    # 创建sad情感，label_id为1，训练集的MELD数据加载器
    dataloader = create_emotion_dataloader(
        data_dir='Data',
        dataset_name='MELD',
        emotion='sad',
        label_id=1,
        split='train',  # MELD需要指定split
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    print(f"DataLoader创建成功!")
    print(f"数据集大小: {len(dataloader.dataset)}")

    # 打印数据集信息
    dataset_info = dataloader.dataset.get_info()
    print(f"\n数据集信息:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60 + "\n")


def example3_multiple_dataloaders():
    """示例3: 批量创建多个DataLoader"""
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
    dataloaders = create_multiple_dataloaders(
        data_dir='Data',
        dataset_name='MOSEI',
        emotion_label_pairs=emotion_label_pairs,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )

    print(f"成功创建 {len(dataloaders)} 个DataLoader:")
    for key, dataloader in dataloaders.items():
        print(f"  {key}: {len(dataloader.dataset)} 个样本")

    print("\n" + "=" * 60 + "\n")


def example4_all_splits_meld():
    """示例4: 为MELD创建所有split的DataLoader"""
    print("=" * 60)
    print("示例4: 为MELD创建train/dev/test三个DataLoader")
    print("=" * 60)

    # 为anger情感创建所有split的DataLoader
    dataloaders = create_all_splits_dataloaders(
        data_dir='Data',
        emotion='anger',
        label_id=2,
        batch_size=32,
        num_workers=4
    )

    print(f"成功创建 {len(dataloaders)} 个DataLoader:")
    for split, dataloader in dataloaders.items():
        print(f"  {split}: {len(dataloader.dataset)} 个样本, {len(dataloader)} 个批次")

    print("\n" + "=" * 60 + "\n")


def example5_custom_collate():
    """示例5: 使用自定义collate函数"""
    print("=" * 60)
    print("示例5: 使用自定义collate函数")
    print("=" * 60)

    dataloader = create_emotion_dataloader(
        data_dir='Data',
        dataset_name='MOSEI',
        emotion='happy',
        label_id=0,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate_fn  # 使用自定义collate函数
    )

    print(f"DataLoader创建成功!")
    print("使用了自定义collate函数来整理batch数据")

    # 获取一个batch
    for batch in dataloader:
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

    # 创建训练和验证DataLoader
    train_loader = create_emotion_dataloader(
        data_dir='Data',
        dataset_name='MELD',
        emotion='happy',
        label_id=0,
        split='train',
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    dev_loader = create_emotion_dataloader(
        data_dir='Data',
        dataset_name='MELD',
        emotion='happy',
        label_id=0,
        split='dev',
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

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

        # 验证阶段
        print("  Validating...")
        for batch_idx, batch in enumerate(dev_loader):
            # 这里进行验证
            # val_loss = model(batch)

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

    dataloader = create_emotion_dataloader(
        data_dir='Data',
        dataset_name='MOSEI',
        emotion='surprise',
        label_id=4,
        batch_size=16,
        transform=my_transform  # 应用转换函数
    )

    print(f"DataLoader创建成功，已应用自定义transform函数")
    print(f"数据集大小: {len(dataloader.dataset)}")

    print("\n" + "=" * 60 + "\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("情感数据集DataLoader使用示例")
    print("=" * 60 + "\n")

    # 运行所有示例（注意：需要实际的pkl文件才能运行）
    print("注意：以下示例需要实际的pkl文件才能正常运行")
    print("请确保Data目录下有对应格式的pkl文件\n")

    try:
        example1_single_mosei_dataloader()
    except Exception as e:
        print(f"示例1出错: {e}\n")

    try:
        example2_single_meld_dataloader()
    except Exception as e:
        print(f"示例2出错: {e}\n")

    try:
        example3_multiple_dataloaders()
    except Exception as e:
        print(f"示例3出错: {e}\n")

    try:
        example4_all_splits_meld()
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

    print("\n所有示例运行完毕!")
