from core import zeroshotExpander
import numpy as np
import torch
import torch.nn.functional as F

def getTransitionProb(x1, x2, num_emotions=6):
    prob_sum = np.sum(x1 * x2)
    x1_x2 = prob_sum/np.sum(x2)/num_emotions # p(x1|x2)
    x2_x1 = prob_sum/np.sum(x1)/num_emotions # p(x2|x1)
    return x1_x2, x2_x1


# get ex trans matrix
def getTransitionMatrix(ex_au):
    num_exs = ex_au.shape[0]
    num_aus = ex_au.shape[1]
    trans_ex = np.zeros((num_exs, num_exs))
    self_connection = np.identity(num_exs)
    b = trans_ex
    for i in range(num_exs - 1):
        for j in range(i + 1, num_exs):
            y1, y2 = getTransitionProb(ex_au[i], ex_au[j])
            b[i][j] = y1
            b[j][i] = y2
    for i in range(num_exs):
        trans_ex[i] = b[i] / np.sum(b[i])
    trans_ex = trans_ex + self_connection
    return trans_ex

def getclassembedding(au_embedding, ex_au):
    num_exs = ex_au.shape[0]
    num_aus = ex_au.shape[1]
    class_vectors = []
    for ex in range(num_exs):
        vector = torch.zeros_like(au_embedding[0])
        cnt = 0
        for au in range(num_aus):
            vector += au_embedding['AU' + str(au + 1)] * ex_au[ex][au]
            cnt += 1
        vector = vector / cnt
        class_vectors.append(vector)
    class_vectors = torch.stack(class_vectors)
    class_vectors = F.normalize(class_vectors)
    return class_vectors

def l2_loss(a, b):
    return ((a - b)**2).sum() / (len(a) * 2)
def mask_l2_loss(a, b, mask):
    return l2_loss(a*mask, b*mask)

num_emotions = 6
num_aus = 23
au_given_emotion = np.random.rand(num_emotions, num_aus)
trans = getTransitionMatrix(au_given_emotion)
hidden_layers = 'd512,d1024,d512,d' # 需要写进yaml配置文件
in_channels = 768 #语义描述的维度，也就是AU语义特征的维度
out_channels = 768 #样本特征维度，与分类器权重维度相同
model_zeroshot = zeroshotExpander(n=num_emotions, edges=trans, in_channels=in_channels, out_channels=out_channels, hidden_layers=hidden_layers)
au_embeddings = torch.load('/media/sda/wf/openMM/codes_v251119_2/materials/au_embedding.pt')
class_embeddings = getclassembedding(au_embeddings, au_given_emotion)
output_vectors = model_zeroshot(class_embeddings)
num_seen = 2
num_unseen = num_emotions - num_seen
fc_vectors = torch.randn(num_emotions, out_channels) #seen分类器的权重
seen_mask = torch.zeros(num_emotions)
seen_mask[:num_seen] = 1.0
loss = mask_l2_loss(output_vectors, fc_vectors, seen_mask)#tlist[:n_train]

