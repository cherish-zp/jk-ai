from transformers import BertModel,BertConfig
import torch

DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")

#加载预训练模型
# pretrained = BertModel.from_pretrained(r"E:\PycharmProjects\demo_7\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f").to(DEVICE)
# pretrained.embeddings.position_embeddings = torch.nn.Embedding(1024,768).to(DEVICE)
config = BertConfig.from_pretrained(r"/Users/zhangpeng/code_bigmodel/jk-ai/00-model/BAAI/bge-m3/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")

config.max_position_embeddings = 1024
config.hidden_size = 768  # 确保隐藏层大小为768

print(config)

#使用配置文件初始化模型
pretrained = BertModel(config).to(DEVICE)
print(pretrained)
#定义下游任务
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768,10)
    def forward(self,input_ids,attention_mask,token_type_ids):
        #冻结预训练模型权重
        # with torch.no_grad():
        #     out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        ## 1024 从 512 tokens 变了 ，需要进行全量微调
        out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 增量模型参与训练
        out = self.fc(out.last_hidden_state[:,0])



        return out

