from datasets import load_dataset
from tqdm import tqdm
import itertools
import os
import json

# 配置：指定保存目录和文件名
output_dir = "/Users/zhangpeng/code_bigmodel/jk-ai/04-HuggingFace/data/redpajama-book-refined-by-data-juicer"
output_file = os.path.join(output_dir, "redpajama_100.json")
os.makedirs(output_dir, exist_ok=True)

# 流式加载数据集
streaming_dataset = load_dataset(
    path="datajuicer/redpajama-book-refined-by-data-juicer",
    split="train",
    streaming=True
)

# 收集前100条样本
samples = []
for sample in tqdm(itertools.islice(streaming_dataset, 100), total=1, desc="Downloading"):
    samples.append(sample)

# 保存为标准 JSON 文件（整个是一个 list）
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"\n✅ 已成功保存 100 条数据到: {output_file}")