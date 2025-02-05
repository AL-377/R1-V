# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import json

from datasets import load_dataset, load_from_disk, Dataset
from transformers import Qwen2VLForConditionalGeneration
from PIL import Image
import datasets

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    min_image_size: Optional[int] = field(
        default=28,
        metadata={"help": "Minimum size for image dimensions"},
    )
    resize_method: Optional[str] = field(
        default="bicubic",
        metadata={"help": "Method to use for image resizing (bicubic, bilinear, nearest)"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has \boxed{} tags
                sol_match = re.search(r'\\boxed{(.*?)}', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has \boxed{} tags
                content_match = re.search(r'\\boxed{(.*?)}', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has the required format:
    - Must have <Thought> tags with content
    - Must have <Output> tags with both explanation and \boxed{answer}
    """
    pattern = r"<Thought>.*?</Thought>\s*<Output>.*?\\boxed{.*?}</Output>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <Thought> </Thought> and <Output> </Output> tags, respectively, i.e., "
    "<Thought> reasoning process here </Thought><Output> answer here </Output>"
)


def preprocess_image(image, min_size=28):
    """
    预处理图片，确保图片尺寸不小于最小要求
    Args:
        image: PIL Image对象
        min_size: 最小尺寸要求
    Returns:
        处理后的PIL Image对象
    """
    width, height = image.size
    
    # 如果图片任一维度小于最小尺寸，进行放大
    if width < min_size or height < min_size:
        # 计算需要的缩放比例
        scale = max(min_size/width, min_size/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 使用BICUBIC算法进行放大
        image = image.resize((new_width, new_height), Image.BICUBIC)
    
    return image


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <Thought> </Thought> and provide a complete answer sentence in <Output> </Output> tags, where the final answer should be enclosed in \\boxed{{}} tags. Example: '<Thought>Here I am asked to ..</Thought><Output>The area of the triangle is \\boxed{{25}} square meters</Output>'"

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }

    # Load the dataset
    if script_args.dataset_name.endswith('.json'):
        # 如果是本地JSON文件
        with open(script_args.dataset_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        for item in data:
            # 处理图片路径并加载图片
            image_path = item['images'][0] if item.get('images') else None
            try:
                # 直接加载并预处理图片数据
                image = Image.open(image_path)
                image = preprocess_image(image, min_size=script_args.min_image_size)  # 确保最小尺寸为28
            except Exception as e:
                print(f"Error loading/processing image {image_path}: {e}")
                continue
            
            # 获取对话内容
            conversations = item.get('conversations', [])
            if len(conversations) >= 2:  # 确保至少有一问一答
                human_msg = next((conv['value'] for conv in conversations if conv['from'] == 'human'), '')
                gpt_msg = next((conv['value'] for conv in conversations if conv['from'] == 'gpt'), '')
                
                # 从human message中提取问题（移除<image>标记）
                problem = human_msg.replace('<image>\n', '').strip()
                
                processed_data.append({
                    'image': image,  # 直接存储图片对象
                    'problem': problem,
                    'solution': gpt_msg,
                    'original_question': problem,
                    'original_answer': gpt_msg,
                })
        
        # 创建数据集，指定image特征为Image类型
        features = datasets.Features({
            'image': datasets.Image(),
            'problem': datasets.Value('string'),
            'solution': datasets.Value('string'),
            'original_question': datasets.Value('string'),
            'original_answer': datasets.Value('string'),
        })
        
        dataset = Dataset.from_list(processed_data, features=features)
        
        # 先进行数据转换
        if "image" in dataset.features:
            print("has image in dataset")
            dataset = dataset.map(make_conversation_image)
        else:
            print("no image in dataset")
            dataset = dataset.map(make_conversation)
            dataset = dataset.remove_columns("messages")
            
        # 然后再进行训练集和测试集的分割
        splits = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = {
            script_args.dataset_train_split: splits['train'],
            script_args.dataset_test_split: splits['test']
        }
    else:
        # 原有的HuggingFace数据集加载逻辑
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        
        if "image" in dataset[script_args.dataset_train_split].features:
            print("has image in dataset")
            dataset = dataset.map(make_conversation_image)
        else:
            print("no image in dataset")
            dataset = dataset.map(make_conversation)
            dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2VLGRPOTrainer


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
