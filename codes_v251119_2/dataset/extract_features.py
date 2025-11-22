#!/usr/bin/env python
"""
Utterance-Level 特征提取脚本
使用 RoBERTa/HuBERT/ViT-Base 提取固定维度特征

使用方法：
1. 编辑 extraction_settings.json 配置文件
2. 运行：python extract_features.py
"""

import os
import json
import logging
import pandas as pd
from tqdm import tqdm
import pickle
from datetime import datetime
from collections import defaultdict

from feature_extractor import create_extractor

import sys
# 添加父目录到路径
cur_file_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(cur_file_dir, 'logs')
print('日志目录:', log_dir)
os.makedirs(log_dir, exist_ok=True)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'{log_dir}/extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


emotion_standard_mapping = {
    'joy': 'joy',
    'happy': 'happy', 'happiness': 'happy',
    'sadness': 'sad', 'sad': 'sad',
    'anger': 'anger', 'angry': 'anger',
    'surprise': 'surprise', 'surprised': 'surprise',
    'disgust': 'disgust', 'disgusted': 'disgust',
    'fear': 'fear', 'fearfull': 'fear',
    'neutral': 'neutral', 
}


def extract_mosei(config):
    """提取 MOSEI 数据集特征"""
    logger.info("\n" + "="*60)
    logger.info("开始提取 MOSEI 数据集")
    logger.info("="*60)

    mosei_config = config['mosei']
    base_dir = mosei_config['base_dir']
    label_file = mosei_config['label_file']
    output_dir = mosei_config['output_dir']

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建提取器
    extractor = create_extractor(config_file)

    # 读取标签文件
    df = pd.read_csv(label_file)
    logger.info(f"数据集总样本数: {len(df)}")

    # 情感映射（支持 angry/anger, surprise/surprised 等变体）
    emotion_mapping = {
        'happy': 0, 
        'sad': 1, 
        'anger': 2, 
        'surprise': 3, 
        'disgust': 4, 
        'fear': 5, 
        'neutral': 6}

    # 按情感分组存储
    emotion_data = defaultdict(list)
    success_count = 0
    fail_count = 0

    # 提取特征
    for index, row in tqdm(df.iterrows(), total=len(df), desc="提取特征", ncols=80):
        video_id = row['video_id']
        clip_id = str(row['clip_id'])
        emotion = row['voted_emotion'].lower()
        label = emotion_mapping[emotion]
        emotion = emotion_standard_mapping.get(emotion, emotion)
        text = row['text']

        # 文件路径
        video_path = os.path.join(base_dir, 'video', video_id, f"{clip_id}.mp4")
        audio_path = os.path.join(base_dir, 'audio', video_id, f"{clip_id}.wav")

        # 检查文件是否存在
        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            fail_count += 1
            continue

        sample_id = f"{video_id}_{clip_id}"
        try:
            features = extractor.extract_multimodal_features(text, audio_path, video_path)

            sample_data = {
                'text_features': features['text_features'],
                'audio_features': features['audio_features'],
                'video_features': features['video_features'],
                'au_features': features['au_features'],
                'label': label,
                'emotion': emotion,
                'sample_id': sample_id
            }

            emotion_data[emotion].append(sample_data)
            success_count += 1

        except Exception as e:
            logger.warning(f"样本 {sample_id} 提取失败: {str(e)}")
            fail_count += 1

    # 保存特征文件
    stats = {}
    for emotion, samples in emotion_data.items():
        if len(samples) > 0:
            output_file = os.path.join(output_dir, f"MOSEI{emotion}label{label}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(samples, f)
            stats[emotion] = len(samples)
            logger.info(f"✓ {emotion}: {len(samples)} 样本 -> {output_file}")

    logger.info("\n" + "="*60)
    logger.info("MOSEI 特征提取完成")
    logger.info(f"成功: {success_count} | 失败: {fail_count}")
    logger.info("="*60 + "\n")

    return stats


def extract_meld(config):
    """提取 MELD 数据集特征"""
    logger.info("\n" + "="*60)
    logger.info("开始提取 MELD 数据集")
    logger.info("="*60)

    meld_config = config['meld']
    base_dir = meld_config['base_dir']
    output_dir = meld_config['output_dir']
    split_mode = meld_config.get('split', 'all')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建提取器
    extractor = create_extractor(config_file)

    # 情感映射
    emotion_mapping = {
        'joy': 0, 'sadness': 1, 'anger': 2,
        'surprise': 3, 'disgust': 4, 'fear': 5, 'neutral': 6
    }

    # 确定要处理的splits
    if split_mode == 'all':
        splits = ['train', 'dev', 'test']
    else:
        splits = [split_mode]

    total_stats = {}

    for split in splits:
        logger.info(f"\n处理 {split} split...")
        split_dir = os.path.join(base_dir, split)
        label_file = os.path.join(split_dir, 'labels.csv')

        if not os.path.exists(label_file):
            logger.warning(f"标签文件不存在: {label_file}")
            continue

        df = pd.read_csv(label_file)
        logger.info(f"{split} 样本数: {len(df)}")

        # 按情感分组存储
        emotion_data = defaultdict(list)
        success_count = 0
        fail_count = 0

        # 提取特征
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"提取 {split}", ncols=80):
            video_name = row['file_id']
            emotion = row['emotion'].lower()
            label = emotion_mapping[emotion]
            emotion = emotion_standard_mapping.get(emotion, emotion)
            text = row['utterance']

            # 文件路径
            video_path = os.path.join(split_dir, 'video', f"{video_name}.mp4")
            audio_path = os.path.join(split_dir, 'audio', f"{video_name}.wav")

            # 检查文件是否存在
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                fail_count += 1
                continue

            sample_id = f"{split}_{video_name}"
            try:
                features = extractor.extract_multimodal_features(text, audio_path, video_path)

                sample_data = {
                    'text_features': features['text_features'],
                    'audio_features': features['audio_features'],
                    'video_features': features['video_features'],
                    'au_features': features['au_features'],
                    'label': label,
                    'emotion': emotion,
                    'sample_id': sample_id
                }

                emotion_data[emotion].append(sample_data)
                success_count += 1

            except Exception as e:
                logger.warning(f"样本 {sample_id} 提取失败: {str(e)}")
                fail_count += 1

        # 保存特征文件
        stats = {}
        for emotion, samples in emotion_data.items():
            if len(samples) > 0:
                output_file = os.path.join(output_dir, f"MELD_{split}{emotion}label{label}.pkl")
                with open(output_file, 'wb') as f:
                    pickle.dump(samples, f)
                stats[emotion] = len(samples)
                logger.info(f"✓ {emotion}: {len(samples)} 样本 -> {output_file}")

        logger.info(f"{split} 完成 - 成功: {success_count} | 失败: {fail_count}")
        total_stats[split] = stats

    logger.info("\n" + "="*60)
    logger.info("MELD 特征提取完成")
    logger.info("="*60 + "\n")

    return total_stats


def extract_iemocap(config):
    """提取 IEMOCAP 数据集特征"""
    import glob
    import soundfile as sf

    logger.info("\n" + "="*60)
    logger.info("开始提取 IEMOCAP 数据集")
    logger.info("="*60)

    iemocap_config = config['iemocap']
    base_dir = iemocap_config['base_dir']
    output_dir = iemocap_config['output_dir']
    temp_video_dir = iemocap_config['temp_video_dir']

    # 创建输出目录和临时目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_video_dir, exist_ok=True)

    # 创建提取器
    extractor = create_extractor(config_file)

    # IEMOCAP 情感映射
    LABEL_MAP = {
        "hap": 0, "sad": 1, "sur": 2, "dis": 3,
        'ang': 4, 'fea': 5, 'fru': 6, 'exc': 7, 'neu': 8,
    }
    # exc 和 hap 合并
    LABEL_MAP["exc"] = 0

    emotion_name_map = {
        0: 'hap', 1: 'sad', 2: 'sur', 3: 'dis',
        4: 'ang', 5: 'fea', 6: 'fru', 7: 'exc', 8: 'neu'
    }

    # 按情感分组存储
    emotion_data = defaultdict(list)
    success_count = 0
    fail_count = 0

    session_ids = list(range(1, 6))  # Session 1-5

    for sess_id in tqdm(session_ids, desc="处理Session", ncols=80):
        sess_path = os.path.join(base_dir, f"Session{sess_id}")
        sess_audio_root = os.path.join(sess_path, "sentences/wav")
        sess_text_root = os.path.join(sess_path, "dialog/transcriptions")
        sess_label_root = os.path.join(sess_path, "dialog/EmoEvaluation")
        sess_video_root = os.path.join(sess_path, "dialog/avi")

        label_paths = glob.glob(os.path.join(sess_label_root, "*.txt"))

        for l_path in tqdm(label_paths, desc=f"Session{sess_id}", ncols=80, leave=False):
            l_name = os.path.basename(l_path)
            dialog_name = l_name.split('.')[0]  # e.g., "Ses01F_impro01"

            # 视频路径
            video_path = os.path.join(sess_video_root, f"{dialog_name}.avi")
            if not os.path.exists(video_path):
                logger.warning(f"视频文件不存在: {video_path}")
                continue

            # 文本路径
            transcripts_path = os.path.join(sess_text_root, l_name)
            with open(transcripts_path, "r") as f:
                transcripts = f.readlines()
                transcripts = {
                    t.split(":")[0]: t.split(":")[1].strip() for t in transcripts
                }

            # 解析标签文件
            with open(l_path, "r") as f:
                label_lines = f.read().split("\n")
                for l in label_lines:
                    if not str(l).startswith("["):
                        continue

                    data = l[1:].split()

                    # 提取信息
                    start_time = float(data[0])
                    end_time = float(data[2][:-1])  # 移除末尾的 ']'
                    utt_id = data[3]  # e.g., "Ses01F_impro01_M000"
                    emo = data[4]

                    # 获取标签
                    label = LABEL_MAP.get(emo, None)
                    if label is None:
                        continue

                    emotion = emotion_name_map[label]

                    # 音频路径
                    wav_folder = utt_id[:-5]
                    wav_name = utt_id + ".wav"
                    audio_path = os.path.join(sess_audio_root, wav_folder, wav_name)

                    if not os.path.exists(audio_path):
                        fail_count += 1
                        continue

                    # 获取文本
                    text_query = utt_id + " [{:08.4f}-{:08.4f}]".format(start_time, end_time)
                    text = transcripts.get(text_query, None)
                    if text is None:
                        # 尝试微调时间戳
                        text_query = utt_id + " [{:08.4f}-{:08.4f}]".format(start_time, end_time + 0.0001)
                        text = transcripts.get(text_query, None)
                        if text is None:
                            text_query = utt_id + " [{:08.4f}-{:08.4f}]".format(start_time + 0.0001, end_time)
                            text = transcripts.get(text_query, None)
                            if text is None:
                                logger.warning(f"文本未找到: {utt_id}")
                                fail_count += 1
                                continue

                    # 提取说话人性别和视频标记
                    speaker_gender = utt_id.split('_')[-1][0]  # 'M' 或 'F'
                    video_marker = utt_id.split('_')[0][-1]    # 'M' 或 'F'

                    sample_id = f"Session{sess_id}_{utt_id}"

                    try:
                        # 预处理视频：切分和裁剪
                        processed_video_path = extractor.preprocess_iemocap_video(
                            video_path=video_path,
                            start_time=start_time,
                            end_time=end_time,
                            speaker_gender=speaker_gender,
                            video_marker=video_marker,
                            temp_dir=temp_video_dir
                        )

                        # 提取多模态特征
                        features = extractor.extract_multimodal_features(
                            text, audio_path, processed_video_path
                        )

                        sample_data = {
                            'text_features': features['text_features'],
                            'audio_features': features['audio_features'],
                            'video_features': features['video_features'],
                            'au_features': features['au_features'],
                            'label': label,
                            'emotion': emotion,
                            'sample_id': sample_id
                        }

                        emotion_data[emotion].append(sample_data)
                        success_count += 1

                        # 删除临时视频文件以节省空间
                        if os.path.exists(processed_video_path):
                            os.remove(processed_video_path)

                    except Exception as e:
                        logger.warning(f"样本 {sample_id} 提取失败: {str(e)}")
                        fail_count += 1
                        # 确保清理临时文件
                        try:
                            if 'processed_video_path' in locals() and os.path.exists(processed_video_path):
                                os.remove(processed_video_path)
                        except:
                            pass

    # 保存特征文件
    stats = {}
    for emotion, samples in emotion_data.items():
        if len(samples) > 0:
            label = samples[0]['label']
            output_file = os.path.join(output_dir, f"IEMOCAP_{emotion}label{label}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(samples, f)
            stats[emotion] = len(samples)
            logger.info(f"✓ {emotion}: {len(samples)} 样本 -> {output_file}")

    logger.info("\n" + "="*60)
    logger.info("IEMOCAP 特征提取完成")
    logger.info(f"成功: {success_count} | 失败: {fail_count}")
    logger.info("="*60 + "\n")

    return stats


def main():
    """主函数 - 完全自动化"""
    """加载配置文件"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    global config_file
    config_file = os.path.join(script_dir, 'extraction_settings.json')
    print('配置文件路径:', config_file)
    config = load_config(config_file)

    logger.info("\n" + "="*60)
    logger.info("Utterance-Level 特征提取系统")
    logger.info(f"配置版本: {config.get('version', 'unknown')}")
    logger.info("="*60)

    # 检查哪些数据集已启用
    enabled_datasets = []
    if config['mosei'].get('enabled', False):
        enabled_datasets.append('mosei')
    if config['meld'].get('enabled', False):
        enabled_datasets.append('meld')
    if config.get('iemocap', {}).get('enabled', False):
        enabled_datasets.append('iemocap')

    if not enabled_datasets:
        logger.error("错误：没有启用任何数据集！")
        logger.error("请在 extraction_settings.json 中设置至少一个数据集的 enabled=true")
        return

    logger.info(f"启用的数据集: {', '.join(enabled_datasets)}\n")

    # 提取特征
    all_stats = {}
    for dataset in enabled_datasets:
        if dataset == 'mosei':
            stats = extract_mosei(config)
            all_stats['mosei'] = stats
        elif dataset == 'meld':
            stats = extract_meld(config)
            all_stats['meld'] = stats
        elif dataset == 'iemocap':
            stats = extract_iemocap(config)
            all_stats['iemocap'] = stats

    # 输出总结
    logger.info("\n" + "="*60)
    logger.info("所有数据集提取完成！")
    logger.info("="*60)
    for dataset, stats in all_stats.items():
        logger.info(f"\n{dataset.upper()}:")
        if isinstance(stats, dict):
            for key, value in stats.items():
                logger.info(f"  {key}: {value if isinstance(value, int) else sum(value.values()) if isinstance(value, dict) else value}")


if __name__ == '__main__':
    main()
