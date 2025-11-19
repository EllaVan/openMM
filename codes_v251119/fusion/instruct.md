📁 MOSEI 数据集输出
output/mosei_utterance_features/
├── MOSEIhappylabel0.pkl          # Happy 情感的所有样本
├── MOSEIsadlabel1.pkl            # Sad 情感的所有样本
├── MOSEIangrylabel2.pkl          # Angry 情感的所有样本（anger会被标准化为angry）
├── MOSEIsurpriselabel3.pkl       # Surprise 情感的所有样本
├── MOSEIdisgustlabel4.pkl        # Disgust 情感的所有样本
├── MOSEIfearlabel5.pkl           # Fear 情感的所有样本
└── MOSEIneutrallabel6.pkl        # Neutral 情感的所有样本

📁 MELD 数据集输出
MELD数据集会按split（train/dev/test）分开保存：

output/meld_utterance_features/
├── MELD_trainhappylabel0.pkl      # Train集的Happy样本
├── MELD_trainsadlabel1.pkl        # Train集的Sad样本
├── MELD_trainangrylabel2.pkl      # Train集的Angry样本
├── MELD_trainsurpriselabel3.pkl   # Train集的Surprise样本
├── MELD_traindisgustlabel4.pkl    # Train集的Disgust样本
├── MELD_trainfearlabel5.pkl       # Train集的Fear样本
├── MELD_trainneutrallabel6.pkl    # Train集的Neutral样本
│
├── MELD_devhappylabel0.pkl        # Dev集的Happy样本
├── MELD_devsadlabel1.pkl          # Dev集的Sad样本
├── MELD_devangrylabel2.pkl        # Dev集的Angry样本
├── MELD_devsurpriselabel3.pkl     # Dev集的Surprise样本
├── MELD_devdisgustlabel4.pkl      # Dev集的Disgust样本
├── MELD_devfearlabel5.pkl         # Dev集的Fear样本
├── MELD_devneutrallabel6.pkl      # Dev集的Neutral样本
│
├── MELD_testhappylabel0.pkl       # Test集的Happy样本
├── MELD_testsadlabel1.pkl         # Test集的Sad样本
├── MELD_testangrylabel2.pkl       # Test集的Angry样本
├── MELD_testsurpriselabel3.pkl    # Test集的Surprise样本
├── MELD_testdisgustlabel4.pkl     # Test集的Disgust样本
├── MELD_testfearlabel5.pkl        # Test集的Fear样本
└── MELD_testneutrallabel6.pkl     # Test集的Neutral样本
总计: 7个情感 × 3个split = 21个文件

📦 每个文件的内容
每个 .pkl 文件是一个列表，包含该情感的所有样本：

[
    {
        'text_features': torch.Tensor([768]),    # RoBERTa
        'audio_features': torch.Tensor([768]),   # HuBERT
        'video_features': torch.Tensor([768]),   # ViT-Base
        'label': 2,                              # 情感ID
        'emotion': 'angry',                      # 标准化后的情感名
        'sample_id': 'video123_clip456'
    },
    {
        'text_features': torch.Tensor([768]),
        'audio_features': torch.Tensor([768]),
        'video_features': torch.Tensor([768]),
        'label': 2,
        'emotion': 'angry',
        'sample_id': 'video789_clip012'
    },
    # ... 更多该情感的样本
]
🔑 关键特点