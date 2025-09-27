from datasets import load_dataset, load_from_disk
import os

# 设置镜像站点（国内访问更稳定）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置数据保存目录
data_dir = r"/Users/zhangpeng/code_bigmodel/jk-ai/04-HuggingFace/data"
dataDirName = "redpajama-book-refined-by-data-juicer"
datasetName = "datajuicer/redpajama-book-refined-by-data-juicer"
data_dir = os.path.join(data_dir, dataDirName)

# 创建目录（如果不存在）
os.makedirs(data_dir, exist_ok=True)

try:
    # 在线加载数据并保存到指定目录
    dataset = load_dataset(
        path=datasetName,
        split="train[:100]",
        cache_dir=data_dir  # 指定缓存目录
    )

    print("数据集信息:")
    print(dataset)
    print(f"\n数据集已保存到: {data_dir}")

    # 保存为可加载的格式
    dataset.save_to_disk(os.path.join(data_dir, "saved_dataset"))
    print("数据集已额外保存为可加载格式")

    # 验证保存的数据
    print("\n验证加载保存的数据:")
    saved_dataset = load_from_disk(os.path.join(data_dir, "saved_dataset"))
    print(saved_dataset)

except Exception as e:
    print(f"错误: {e}")
    print("尝试方案二：使用离线模式或手动下载")