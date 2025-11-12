"""
MELD数据集特征提取脚本
- 文本特征: BERT
- 视频特征: ViLT
- 音频特征: HuBERT
"""

import os
import pandas as pd
import torch
from transformers import (
    BertTokenizer, BertModel,
    Wav2Vec2Processor, HubertModel,
    ViltProcessor, ViltModel
)
import numpy as np
import cv2
import torchaudio
from tqdm import tqdm
import torch.nn.functional as F

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 文件路径配置 ====================
# TODO: 请根据实际情况修改以下路径

# MELD数据集根目录
base_dir = '/path/to/Datasets/MELD/organized/dev'  # TODO: 修改为实际路径

# 数据目录
wav_dir = os.path.join(base_dir, 'audio')
video_dir = os.path.join(base_dir, 'video')
text_dir = os.path.join(base_dir, 'text')
label_file = os.path.join(base_dir, 'label/merged_label_new.csv')  # TODO: 修改标签文件名

# 预训练模型路径
# TODO: 修改为实际的预训练模型路径
BERT_MODEL_PATH = "/path/to/bert-base-uncased"
HUBERT_MODEL_PATH = "/path/to/hubert-base-ls960"
VILT_MODEL_PATH = "/path/to/vilt-b32-mlm"  # ViLT模型路径

# 输出文件路径
save_path = 'meld_multimodal_features_vilt.npy'  # TODO: 可修改输出文件名

# ==================== 加载预训练模型 ====================
print("加载预训练模型...")

# 加载HuBERT音频模型
processor_hubert = Wav2Vec2Processor.from_pretrained(HUBERT_MODEL_PATH)
model_hubert = HubertModel.from_pretrained(HUBERT_MODEL_PATH).to(device)
model_hubert.eval()
print("✓ HuBERT模型加载完成")

# 加载BERT文本模型
tokenizer_bert = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
model_bert = BertModel.from_pretrained(BERT_MODEL_PATH).to(device)
model_bert.eval()
print("✓ BERT模型加载完成")

# 加载ViLT视觉-语言模型
processor_vilt = ViltProcessor.from_pretrained(VILT_MODEL_PATH)
model_vilt = ViltModel.from_pretrained(VILT_MODEL_PATH).to(device)
model_vilt.eval()
print("✓ ViLT模型加载完成")

# ==================== 音频处理函数 ====================
def load_audio(audio_path):
    """
    加载音频文件并进行预处理
    - 重采样到16kHz
    - 转换为单声道
    """
    waveform, sample_rate = torchaudio.load(audio_path)

    # 转换为float32
    waveform = waveform.to(torch.float32)

    # 处理双声道 -> 单声道
    if waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 重采样到16kHz
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampled_waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        )
    else:
        resampled_waveform = waveform

    return resampled_waveform.squeeze(0)


def extract_audio_features(audio_path):
    """
    使用HuBERT提取音频特征
    返回: (1, seq_len, hidden_dim) 的特征向量
    """
    try:
        audio = load_audio(audio_path)
        inputs = processor_hubert(
            audio.squeeze(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model_hubert(**inputs)

        # 返回最后一层隐藏状态
        return outputs.last_hidden_state.cpu().numpy()
    except Exception as e:
        print(f"音频特征提取失败 {audio_path}: {e}")
        return None


# ==================== 文本处理函数 ====================
def extract_text_features(text):
    """
    使用BERT提取文本特征
    返回: (1, seq_len, hidden_dim) 的特征向量
    """
    try:
        inputs = tokenizer_bert(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model_bert(**inputs)

        # 返回最后一层隐藏状态
        return outputs.last_hidden_state.cpu().numpy()
    except Exception as e:
        print(f"文本特征提取失败: {e}")
        return None


# ==================== 视频处理函数 ====================
def extract_video_features(video_path, text=""):
    """
    使用ViLT提取视频特征
    ViLT是视觉-语言模型,可以同时处理图像和文本

    参数:
        video_path: 视频文件路径
        text: 文本信息(可选,用于视觉-语言联合编码)
    返回: (num_frames, seq_len, hidden_dim) 的特征向量
    """
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 设置采样率: 每秒采样5帧
        frames_per_second = 5
        frame_interval = max(1, int(fps / frames_per_second))

        count = 1
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 按间隔采样帧
            if count % frame_interval == 0:
                # 将BGR转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            count += 1

        cap.release()

        if len(frames) == 0:
            print(f"视频无法读取帧: {video_path}")
            return None

        # 使用ViLT处理每一帧
        # ViLT可以接受文本输入,这里可以传入对应的对话文本
        all_features = []

        for frame in frames:
            # ViLT processor需要PIL Image格式
            from PIL import Image
            frame_pil = Image.fromarray(frame)

            # 如果有文本,使用视觉-语言联合编码
            if text and text.strip():
                inputs = processor_vilt(
                    images=frame_pil,
                    text=text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)
            else:
                # 仅使用视觉编码(传入空文本)
                inputs = processor_vilt(
                    images=frame_pil,
                    text="",
                    return_tensors="pt",
                    padding=True
                ).to(device)

            with torch.no_grad():
                outputs = model_vilt(**inputs)

            # 获取序列输出 (包含视觉和文本token的联合表示)
            features = outputs.last_hidden_state.cpu().numpy()
            all_features.append(features)

        # 拼接所有帧的特征: (num_frames, seq_len, hidden_dim)
        video_features = np.concatenate(all_features, axis=0)

        return video_features

    except Exception as e:
        print(f"视频特征提取失败 {video_path}: {e}")
        return None


# ==================== 特征对齐函数 ====================
def align_features(audio_features, text_features, video_features, target_length=512):
    """
    将三个模态的特征对齐到相同长度

    参数:
        audio_features: (1, seq_len, dim)
        text_features: (1, seq_len, dim)
        video_features: (num_frames, seq_len, dim)
        target_length: 目标序列长度

    返回:
        对齐后的特征 (1, target_length, dim)
    """
    # 转换为tensor
    audio_features = torch.tensor(audio_features) if not isinstance(audio_features, torch.Tensor) else audio_features
    text_features = torch.tensor(text_features) if not isinstance(text_features, torch.Tensor) else text_features
    video_features = torch.tensor(video_features) if not isinstance(video_features, torch.Tensor) else video_features

    # 视频特征处理: 先对所有帧求平均
    if video_features.dim() == 3 and video_features.shape[0] > 1:
        video_features = video_features.mean(dim=0, keepdim=True)

    # 对齐函数
    def align_single_feature(feature, target_len):
        if feature.shape[1] < target_len:
            # 填充
            padding = target_len - feature.shape[1]
            feature_padded = F.pad(feature, (0, 0, 0, padding))
        elif feature.shape[1] > target_len:
            # 截断
            feature_padded = feature[:, :target_len, :]
        else:
            feature_padded = feature
        return feature_padded

    # 对齐三个模态
    audio_aligned = align_single_feature(audio_features, target_length)
    text_aligned = align_single_feature(text_features, target_length)
    video_aligned = align_single_feature(video_features, target_length)

    return audio_aligned, text_aligned, video_aligned


# ==================== 主处理流程 ====================
def main():
    """
    主处理流程:
    1. 读取标签文件
    2. 遍历每个样本
    3. 提取音频、文本、视频特征
    4. 对齐特征并保存
    """
    print("=" * 60)
    print("开始处理MELD数据集")
    print("=" * 60)

    # 检查路径
    if not os.path.exists(label_file):
        print(f"错误: 标签文件不存在 {label_file}")
        print("请在脚本中修改 label_file 路径")
        return

    # 加载CSV标签文件
    df = pd.read_csv(label_file)
    print(f"共找到 {len(df)} 个样本")

    # 情感标签映射
    # TODO: 根据实际的MELD标签修改
    emotion_mapping = {
        'happy': 0,
        'sad': 1,
        'anger': 2,
        'disgust': 3,
        'surprise': 4,
        'fear': 5,
        'neutral': 6  # MELD包含neutral类别
    }

    # 清空已有输出文件
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"删除已存在的输出文件: {save_path}")

    # 统计信息
    success_count = 0
    fail_count = 0

    # 遍历处理每个样本
    for index, row in tqdm(df.iterrows(), desc="处理MELD数据集", total=len(df), ncols=80):
        try:
            # 获取样本信息
            # TODO: 根据实际CSV列名修改
            video_id = row['video_id']
            clip_id = row['clip_id']
            text = row['text']
            voted_emotion = row['voted_emotion']

            # 跳过neutral类别 (如果需要)
            # if voted_emotion == 'neutral':
            #     continue

            # 获取标签
            if voted_emotion not in emotion_mapping:
                print(f"跳过未知情感标签: {voted_emotion}")
                continue

            label = emotion_mapping[voted_emotion]

            # 构建文件路径
            # TODO: 根据实际的文件组织结构修改路径格式
            audio_path = os.path.join(wav_dir, video_id, f"{clip_id}.wav")
            video_path = os.path.join(video_dir, video_id, f"{clip_id}.mp4")

            # 检查文件是否存在
            if not os.path.exists(audio_path):
                print(f"音频文件不存在: {audio_path}")
                fail_count += 1
                continue

            if not os.path.exists(video_path):
                print(f"视频文件不存在: {video_path}")
                fail_count += 1
                continue

            # 提取特征
            audio_features = extract_audio_features(audio_path)
            text_features = extract_text_features(text)
            video_features = extract_video_features(video_path, text=text)  # 传入文本用于ViLT

            # 检查特征提取是否成功
            if audio_features is None or text_features is None or video_features is None:
                print(f"特征提取失败,跳过样本: {video_id}/{clip_id}")
                fail_count += 1
                continue

            # 对齐特征长度
            audio_aligned, text_aligned, video_aligned = align_features(
                audio_features,
                text_features,
                video_features,
                target_length=512
            )

            # 构建样本数据
            sample_data = {
                'audio_features': audio_aligned,
                'text_features': text_aligned,
                'video_features': video_aligned,
                'label': label,
                'video_id': video_id,
                'clip_id': clip_id
            }

            # 追加保存到.npy文件
            with open(save_path, 'ab') as f:
                np.save(f, sample_data)

            success_count += 1

        except Exception as e:
            print(f"处理样本 {index} 时出错: {e}")
            fail_count += 1
            continue

    print("=" * 60)
    print(f"处理完成!")
    print(f"成功: {success_count} 个样本")
    print(f"失败: {fail_count} 个样本")
    print(f"数据已保存到: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
