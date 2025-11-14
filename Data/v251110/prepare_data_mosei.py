# mosei
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, Wav2Vec2Processor, HubertModel, ViTImageProcessor, ViTModel
import numpy as np
import cv2
import soundfile as sf
from tqdm import tqdm
import math
import torchaudio
import subprocess  # 用于调用 OpenFace 命令
import tempfile
import torch.nn.functional as F

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 文件路径
base_dir = '/media/sda/wf/openMM/Datasets/MOSEI'
wav_dir = os.path.join(base_dir, 'audio')
video_dir = os.path.join(base_dir, 'video')
label_file = os.path.join(base_dir, 'label/merged_label_new.csv')
openface_dir = '/media/sda/pingjm/OpenFace-master/build/bin'  # OpenFace 可执行文件路径

# 加载模型
processor_hubert = Wav2Vec2Processor.from_pretrained("/media/sda/pingjm/MTCA/pretraining_model/HuBERT/hubert-base-ls960")
model_hubert = HubertModel.from_pretrained("/media/sda/pingjm/MTCA/pretraining_model/HuBERT/hubert-base-ls960").to(device)
tokenizer_bert = BertTokenizer.from_pretrained("/media/sda/pingjm/MTCA/pretraining_model/BERT/bert-base-uncased")
model_bert = BertModel.from_pretrained("/media/sda/pingjm/MTCA/pretraining_model/BERT/bert-base-uncased").to(device)
feature_extractor_vit = ViTImageProcessor.from_pretrained("/media/sda/pingjm/MTCA/pretraining_model/ViT/vit-base-patch16-224-in21k")
model_vit = ViTModel.from_pretrained("/media/sda/pingjm/MTCA/pretraining_model/ViT/vit-base-patch16-224-in21k").to(device)

# 加载 CSV 文件
df = pd.read_csv(label_file)

# 加载音频文件
def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)  # 加载音频，默认可能是 Float 类型

    # 关键步骤：将 Float 张量转为 Double 张量（torch.float64）
    waveform_double = waveform.to(torch.float32)  # 或 waveform.double()
    waveform_double = torch.tensor(waveform_double).unsqueeze(0)
    # 假设 waveform_double 的形状是 [1, 1, 2, 287200]
    # waveform_double = waveform_double.squeeze(1)  # 去掉多余的维度
    
    # 处理双通道
    if waveform_double.shape[1] == 2:
        waveform_double = waveform_double.mean(dim=1, keepdim=True)

    # 调用重采样函数（此时输入为 Double 类型，匹配函数要求）
    target_sample_rate = 16000
    resampled_waveform = torchaudio.functional.resample(
        waveform_double,
        orig_freq=sample_rate,
        new_freq=target_sample_rate
    )

    # 若后续需要 Float 类型，可再转换回来（可选）
    # resampled_waveform_float = resampled_waveform.to(torch.float32)
    return resampled_waveform.squeeze(0)

# 提取音频特征
def extract_audio_features(audio_path):
    audio = load_audio(audio_path)
    inputs = processor_hubert(audio.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model_hubert(**inputs)
    return outputs.last_hidden_state.cpu().numpy()

# 提取文本特征
def extract_text_features(text):
    inputs = tokenizer_bert(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    return outputs.last_hidden_state.cpu().numpy()

# 计算最长视频帧数
def calculate_max_frames(video_paths, frame_rate=3):
    max_frames = 0
    for video_path in tqdm(video_paths, desc="Calculating max frames"):  # 添加进度条
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_extract = min(total_frames, int((total_frames / fps) * frame_rate))
        max_frames = max(max_frames, frames_to_extract)
        cap.release()
    return max_frames

# 使用 OpenFace 提取人脸特征
def extract_face_features(video_path, video_id, clip_id, selected_frame_indices):
    # 创建一个临时目录用于存放 OpenFace 输出
    with tempfile.TemporaryDirectory() as output_dir:

        # output_path = os.path.join(output_dir, f"{video_id}_{clip_id}.csv")
        output_path = os.path.join(output_dir, f"{clip_id}.csv")
        
        # 调用 OpenFace
        command = f"{openface_dir}/FeatureExtraction -f {video_path} -out_dir {output_dir}"
        subprocess.run(command, shell=True)
        
        # 检查 OpenFace 输出的特征文件
        if os.path.exists(output_path):
            face_features_df = pd.read_csv(output_path)
            
            # 使用指定帧编号的特征数据
            face_features_df = face_features_df[face_features_df['frame'].isin(selected_frame_indices)]
            
            # 检查并替换 NaN 值为 0
            face_features_df = face_features_df.fillna(0)
            
            # 提取 AU (动作单元) 和其他相关特征，按列求平均值
            face_features = face_features_df.iloc[:, 3:]  # 跳过前三列的非特征数据    

            return face_features
        else:
            print(f"OpenFace 无法处理视频 {video_path}")
            return None

# 定义从视频中提取帧并提取特征的函数
def extract_video_features(video_path, video_id, clip_id):
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the video's frames per second

    # Calculate the interval for sampling, aiming for 5 frames per second
    frames_per_second = 5
    frame_interval = int(fps / frames_per_second)

    count = 1
    frames = []
    selected_frame_indices = []  # To store indices of sampled frames
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames at regular intervals
        if count % frame_interval == 0:
            frames.append(frame)
            selected_frame_indices.append(count)  # Record the frame number
        
        count += 1

    cap.release()

    if len(frames) > 0:
        # 使用 ViT 提取视频帧特征
        inputs = feature_extractor_vit(images=frames, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_vit(**inputs)
        video_features = outputs.last_hidden_state.cpu().numpy()

        '''
        # 提取人脸特征并拼接
        face_features = extract_face_features(video_path, video_id, clip_id, selected_frame_indices)

        if face_features is not None:
            # video_features.shape: (10, 197, 768)
            # face_features.shape: (10, 711)

            # 将人脸特征重复到与视频帧特征的长度一致
            # 假设我们为每一帧都附加一个对应的脸部特征
            expanded_face_features = np.expand_dims(face_features, axis=1)  # shape: (10, 1, 711)
            padding_needed = video_features.shape[2] - expanded_face_features.shape[2]  # Calculate the padding needed in the last dimension

            # Pad the array with zeros in the last dimension (axis=2)
            padded_face_features = np.pad(expanded_face_features, ((0, 0), (0, 0), (0, padding_needed)), mode='constant', constant_values=0)

            # 拼接视频特征和人脸特征
            combined_features = np.concatenate([video_features, padded_face_features], axis=1)  # shape: (10, 197 + 1, 768)
 
        else:
            # 生成与 video_features 相同形状的随机 tensor (与视频帧的数量一致)
            expanded_face_features = torch.randn(video_features.shape[0], 1, video_features.shape[2]).cpu().numpy()
            # 拼接视频特征和人脸特征
            combined_features = np.concatenate([video_features, expanded_face_features], axis=1)  # shape: (10, 197 + 1, 768)
        combined_features = np.mean(combined_features, axis=1)
        combined_features = np.expand_dims(combined_features, axis=0)
        return combined_features
        '''
        video_features = video_features.mean(axis=1)
        video_features = np.expand_dims(video_features, axis=0)
        return video_features
    return None

# 构建数据集
save_path = 'mosi_multimodal_features_with_openface_v3_mosei_new.npy'

emotion_mapping = {'happy': 0, 'sad': 1, 'anger': 2, 'disgust': 3, 'surprise': 4, 'fear': 5}

# 清空已有文件
if os.path.exists(save_path):
    os.remove(save_path)

video_paths = [os.path.join(video_dir, row['video_id'], f"{row['clip_id']}.mp4") for _, row in df.iterrows() if os.path.exists(os.path.join(video_dir, row['video_id'], f"{row['clip_id']}.mp4"))]

for index, row in tqdm(df.iterrows(), desc="Processing MOSEI dataset", total=len(df), ncols=70):
    voted_emotion = row['voted_emotion']
    if voted_emotion !='neural':
    
        video_id = row['video_id']
        clip_id = row['clip_id']
        text = row['text']
        label = emotion_mapping[row['voted_emotion']]
        
        audio_path = os.path.join(wav_dir, video_id, f"{clip_id}.wav")
        video_path = os.path.join(video_dir, video_id, f"{clip_id}.mp4")
        
        if os.path.exists(audio_path) and os.path.exists(video_path):
            audio_features = extract_audio_features(audio_path)
            text_features = extract_text_features(text)
            video_features = extract_video_features(video_path, video_id, clip_id)

            audio_features = torch.tensor(audio_features) if not isinstance(audio_features, torch.Tensor) else audio_features
            text_features = torch.tensor(text_features) if not isinstance(text_features, torch.Tensor) else text_features
            video_features = torch.tensor(video_features) if not isinstance(video_features, torch.Tensor) and video_features is not None else video_features

            # 目标长度
            target_length = 512

            # 对齐 text_features 和 video_features
            # 使用填充 (pad) 或截断 (slice) 来对齐长度

            # 对齐 audio_features
            if audio_features.shape[1] < target_length:
                padding = target_length - audio_features.shape[1]
                audio_features_padded = F.pad(audio_features, (0, 0, 0, padding))  # 在最后一维填充
            elif audio_features.shape[1] > target_length:
                audio_features_padded = audio_features[:, :target_length, :]  # 截断到目标长度
            else:
                audio_features_padded = audio_features

            # 对齐 text_features
            if text_features.shape[1] < target_length:
                padding = target_length - text_features.shape[1]
                text_features_padded = F.pad(text_features, (0, 0, 0, padding))  # 在最后一维填充
            elif text_features.shape[1] > target_length:
                text_features_padded = text_features[:, :target_length, :]  # 截断到目标长度
            else:
                text_features_padded = text_features

            # 对齐 video_features
            if video_features.shape[1] < target_length:
                padding = target_length - video_features.shape[1]
                video_features_padded = F.pad(video_features, (0, 0, 0, padding))  # 在最后一维填充
            elif video_features.shape[1] > target_length:
                video_features_padded = video_features[:, :target_length, :]  # 截断到目标长度
            else:
                video_features_padded = video_features
            # import pdb
            # pdb.set_trace()
            sample_data = {
                'audio_features': audio_features_padded,
                'text_features': text_features_padded,
                'video_features': video_features_padded,
                'label': label
            }

            # 追加写入 .npy 文件
            with open(save_path, 'ab') as f:
                np.save(f, sample_data)

        else:
            print(f"音频文件 {audio_path} 或者视频文件 {video_path} 不存在，跳过该数据。")

print(f"数据集已保存到{save_path}")