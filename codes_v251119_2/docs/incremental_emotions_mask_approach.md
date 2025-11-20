# 增量情绪处理：Mask方案

## 概述

实现了基于mask的增量情绪处理机制，允许模型在持续学习过程中动态管理活跃的情绪集合。

## 核心特性

### 1. 固定大小矩阵 + 动态掩码

- **矩阵大小**: 始终保持 `[7, 23]` (7个情绪 × 23个AUs)
- **动态激活**: 使用 `active_mask` 布尔张量跟踪哪些情绪当前活跃
- **输出过滤**: `forward()` 返回 `[batch, num_active]` 而非 `[batch, 7]`

### 2. 关键方法

#### `add_emotion(emotion_name: str, global_idx: int)`
激活一个新情绪，用于task开始时或遇到unseen情绪时。

```python
# 示例：Task 0开始时激活seen和unseen情绪
matrix.add_emotion("happy", 0)
matrix.add_emotion("sad", 1)
matrix.add_emotion("surprise", 2)  # unseen
matrix.add_emotion("disgust", 3)   # unseen
```

**参数**:
- `emotion_name`: 情绪名称（用于日志）
- `global_idx`: 在全局emotion_id_mapping中的索引（0-6）

**行为**:
- 如果情绪已激活，跳过（幂等操作）
- 更新 `active_mask[global_idx] = True`
- 维护 `emotion_id_to_local_idx` 和 `local_idx_to_emotion_id` 映射
- 增加 `num_active_emotions` 计数

#### `get_active_emotions() -> list`
返回当前激活的情绪全局ID列表（按local_idx排序）。

```python
active_ids = matrix.get_active_emotions()
# Task 0后: [0, 1, 2, 3] (happy, sad, surprise, disgust)
# Task 1后: [0, 1, 2, 3, 4, 5] (+ anger, fear)
```

#### `reset_active_emotions()`
重置所有激活状态（用于测试或重新开始）。

### 3. Forward Pass 行为变化

**之前**: 返回 `[batch, 7]` - 所有情绪的logits

**现在**: 返回 `[batch, num_active]` - 仅活跃情绪的logits

```python
# Task 0: 4个活跃情绪
au_probs = torch.rand(32, 23)  # batch=32
emo_logits = matrix(au_probs)
# 输出shape: [32, 4]

# Task 1: 6个活跃情绪
emo_logits = matrix(au_probs)
# 输出shape: [32, 6]

# Task 2: 7个活跃情绪
emo_logits = matrix(au_probs)
# 输出shape: [32, 7]
```

**特殊情况**: 如果没有激活任何情绪，返回 `[batch, 0]`

### 4. 标签映射

**重要**: 输出logits的索引对应 **local indices**，不是global emotion IDs！

```python
# Task 0
active_emotions = [0, 1, 2, 3]  # [happy, sad, surprise, disgust]
emo_logits shape: [B, 4]

# 如果样本标签是 disgust (global_id=3)
local_idx = matrix.emotion_id_to_local_idx[3]  # = 3
# 使用 local_idx 作为 cross_entropy 的目标

loss = F.cross_entropy(emo_logits, torch.tensor([local_idx]))
```

### 5. 保存和加载

增量学习状态会自动保存和恢复：

- `active_mask`: 布尔掩码
- `num_active_emotions`: 活跃情绪数
- `emotion_id_to_local_idx`: 全局ID → 局部索引映射
- `local_idx_to_emotion_id`: 局部索引 → 全局ID映射

```python
# 保存
matrix.save("checkpoint.npz")
# 输出: 活跃情绪数: 4/7

# 加载
matrix.load("checkpoint.npz")
# 自动恢复增量学习状态
```

## 使用示例

### 典型的持续学习流程

```python
# 初始化矩阵
matrix = LearnableAUEMOMatrix(
    num_aus=23,
    num_emotions=7,
    prior_p_au_given_emo=prior_matrix,
    device='cuda:0'
)

# ===== Task 0 =====
# 激活seen和unseen情绪
for emo_name, emo_id in task0_emotions.items():
    matrix.add_emotion(emo_name, emo_id)

# 训练
for epoch in range(num_epochs):
    for batch in dataloader:
        au_probs = au_encoder(batch)
        emo_logits = matrix(au_probs)  # [B, 4]

        # 将全局标签映射到局部索引
        local_labels = torch.tensor([
            matrix.emotion_id_to_local_idx[global_id.item()]
            for global_id in batch['emotion_labels']
        ])

        loss = F.cross_entropy(emo_logits, local_labels)
        loss.backward()
        optimizer.step()

# ===== Task 1 =====
# 激活新情绪（已激活的会自动跳过）
for emo_name, emo_id in task1_emotions.items():
    matrix.add_emotion(emo_name, emo_id)

# 继续训练...
```

## 优势

1. **简单实现**: 只需添加mask和映射管理，不改变矩阵结构
2. **EWC友好**: 固定大小矩阵，Fisher信息矩阵维度不变
3. **全局语义**: 每个情绪始终对应相同的矩阵行
4. **状态持久化**: 增量学习状态完全可保存/恢复

## 注意事项

1. **标签转换必需**: 训练时必须将global emotion ID转换为local index
2. **输出维度动态**: 每个task的输出维度会变化，需要相应调整loss计算
3. **幂等激活**: 多次激活同一情绪是安全的（会跳过）
4. **内存开销**: 保持7×23矩阵，即使只用到部分情绪（但开销可忽略）

## 测试

运行测试脚本验证功能：

```bash
python test_incremental_emotions.py
```

测试覆盖：
- ✓ 增量激活情绪（Task 0 → Task 1 → Task 2）
- ✓ Forward pass输出维度正确
- ✓ 重复激活会跳过
- ✓ 保存和加载状态
- ✓ 重置功能
