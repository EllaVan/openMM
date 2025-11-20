import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roberta_path = '/media/sda/wf/openMM/materials_task/roberta-base'
tokenizer_bert =  AutoTokenizer.from_pretrained(roberta_path)
model_bert = AutoModel.from_pretrained(roberta_path).to(device)
model_bert.eval()

def extract_text_features(text):
    inputs = tokenizer_bert(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model_bert(**inputs)
        features = outputs.last_hidden_state[0, 0, :]  # [768]
    return features.cpu()

au_description = {}
with open('/media/sda/wf/openMM/codes_v251119_2/materials/AU_action.txt', "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  # 去掉列表中每一个元素的换行符
        terms = line.split(':')
        au_description[terms[0]] = terms[1]
au_name = list(au_description)
au_vectors = []
au_embedding = {}
for i in range(len(au_name)):
    au_vectors.append(extract_text_features(au_description[au_name[i]]))
au_vectors = torch.stack(au_vectors)
au_vectors = F.normalize(au_vectors)
for i in range(len(au_name)):
    au_embedding[au_name[i]] = au_vectors[i]
with open('/media/sda/wf/openMM/codes_v251119_2/materials/au_embedding.pt', 'wb') as f:
    torch.save(au_vectors, f)
end = 1