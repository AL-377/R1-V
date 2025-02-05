import json
import os
from datasets import Dataset, Image, Features, Value, DatasetDict
from tqdm import tqdm
from PIL import Image as PILImage

def create_hf_dataset(json_path, output_dir="processed_dataset"):
    """
    将ShareGPT格式的JSON转换为HuggingFace dataset格式
    
    Args:
        json_path: ShareGPT格式JSON文件的路径
        output_dir: 处理后数据的输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理数据
    processed_data = {
        'image': [],
        'problem': [],
        'solution': []
    }
    
    print("Processing data...")
    for item in tqdm(data):
        try:
            # 获取图片路径
            image_path = item['images'][0]
            # 读取图片
            image = PILImage.open(image_path)
            
            # 获取对话内容
            human_msg = next((conv['value'] for conv in item['conversations'] if conv['from'] == 'human'), '')
            gpt_msg = next((conv['value'] for conv in item['conversations'] if conv['from'] == 'gpt'), '')
            
            # 处理问题（移除<image>标记）
            problem = human_msg.replace('<image>\n', '').strip()
            
            # 添加到处理后的数据中
            processed_data['image'].append(image)
            processed_data['problem'].append(problem)
            processed_data['solution'].append(gpt_msg)
            
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    # 创建Dataset
    features = Features({
        'image': Image(),
        'problem': Value('string'),
        'solution': Value('string')
    })
    
    dataset = Dataset.from_dict(
        processed_data,
        features=features
    )

    # # 分割训练集和验证集
    # dataset = dataset.train_test_split(test_size=0.1, seed=42)
  
    
    return dataset

def push_to_hub(dataset, repo_name):
    """
    将数据集上传到Hugging Face Hub
    
    Args:
        dataset: 处理好的数据集
        repo_name: Hugging Face Hub上的仓库名称
    """
    dataset.push_to_hub(repo_name)

if __name__ == "__main__":
    # 配置参数
    json_path = "/map-vepfs/ljt/R1-V/data/data_0_3/virgo_refined_rl_sample_sharegpt_all_acc.json"
    your_hf_repo = "Open-MMO1/virgo_qvqbo16_acc_0_3"  # 替换为你的HF仓库名
    
    # 创建数据集
    dataset = create_hf_dataset(json_path)
    
    # 上传到Hugging Face Hub
    push_to_hub(dataset, your_hf_repo)
    
    print(f"Dataset has been processed and uploaded to {your_hf_repo}")
