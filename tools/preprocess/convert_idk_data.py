#!/usr/bin/env python3
import json
import os
from pathlib import Path

def convert_conversations_to_messages(conversations):
    """将conversations格式转换为messages格式"""
    messages = []
    
    for conv in conversations:
        if conv["from"] == "human":
            # 处理用户消息，保持<image>标签
            content = conv["value"]
            messages.append({
                "role": "user", 
                "content": content
            })
        elif conv["from"] == "gpt":
            # 处理助手消息
            messages.append({
                "role": "assistant",
                "content": conv["value"]
            })
    
    return messages

def convert_image_path(image_path, coco_base_path):
    """转换图片路径"""
    # 从 "MSCOCO/images/train2014/COCO_train2014_000000393224.jpg" 
    # 提取文件名 "COCO_train2014_000000393224.jpg"
    filename = os.path.basename(image_path)
    
    # 根据文件名确定子目录
    if "train2014" in filename:
        subdir = "train2014"
    elif "val2014" in filename:
        subdir = "val2014"
    elif "test2014" in filename:
        subdir = "test2014"
    else:
        # 默认使用train2014
        subdir = "train2014"
    
    # 构建完整路径
    full_path = os.path.join(coco_base_path, subdir, filename)
    return full_path

def convert_vqav2_idk_to_messages(question, answer_list):
    """将VQAv2-IDK格式转换为messages格式"""
    messages = []
    
    # 用户问题
    messages.append({
        "role": "user",
        "content": f"<image>{question}"
    })
    
    # # 选择最常见的答案作为回复，如果有"not sure"或"none"等不确定答案则优先使用
    # uncertainty_answers = ["not sure", "none", "i don't know", "idk", "unknown", "unclear"]
    
    # # 统计答案频率
    answer_count = {}
    for ans in answer_list:
        ans_lower = ans.lower().strip()
        answer_count[ans_lower] = answer_count.get(ans_lower, 0) + 1
    
    # # 优先选择不确定性答案
    # selected_answer = None
    # for uncertainty in uncertainty_answers:
    #     if uncertainty in answer_count:
    #         selected_answer = f"I'm not sure about that. The answer could be {', '.join(list(set([ans for ans in answer_list if ans.lower().strip() != uncertainty]))[:3])}."
    #         break
    
    # # 如果没有不确定性答案，选择最频繁的答案
    # if not selected_answer:
    #     most_common = max(answer_count.items(), key=lambda x: x[1])
    #     selected_answer = f"Based on the image, I believe the answer is: {most_common[0]}."
    most_common = max(answer_count.items(), key=lambda x: x[1])
    selected_answer = most_common[0]
    messages.append({
        "role": "assistant", 
        "content": selected_answer
    })
    
    return messages

def convert_idk_data(input_file, output_file, coco_base_path):
    """转换IDK数据格式"""
    print(f"正在转换 {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    # 判断数据格式类型
    if isinstance(data, list):
        # VQAv2-IDK格式 (直接是列表)
        print("检测到VQAv2-IDK格式")
        for item in data:
            # 转换图片路径
            image_path = convert_image_path(item["image"], coco_base_path)
            
            # 转换问答格式
            messages = convert_vqav2_idk_to_messages(item["question"], item["keywords"])
            
            # 构建新的数据项
            converted_item = {
                "messages": messages,
                "images": image_path,
                "solution": messages[-1]["content"]  # 将最后一条assistant回答放到solution字段
            }
            
            converted_data.append(converted_item)
            
    elif isinstance(data, dict) and "data" in data:
        # 原始IDK格式 (有data字段)
        print("检测到原始IDK格式")
        for item in data["data"]:
            # 转换图片路径
            image_path = convert_image_path(item["image"], coco_base_path)
            
            # 转换对话格式
            messages = convert_conversations_to_messages(item["conversations"])
            
            # 构建新的数据项
            solution = messages[-1]["content"]
            del messages[-1]
            converted_item = {
                "messages": messages,
                "images": image_path,
                "solution": solution  # 将最后一条assistant回答放到solution字段
            }
            
            converted_data.append(converted_item)
    else:
        print(f"未知的数据格式: {type(data)}")
        return
    
    # 保存转换后的数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成，共处理 {len(converted_data)} 条数据")

def main():
    # 设置路径
    input_dir = "/data/dlf/code/Field-Fidelity/data/idk/data"
    output_dir = "/data/dlf/code/Field-Fidelity/data/idk/data_format"
    coco_base_path = "/data/dlf/code/Field-Fidelity/data/coco"
    
    # 要转换的文件列表
    files_to_convert = [
        ("idk_train.json", "idk_train_formatted.jsonl"),
        ("idk_val.json", "idk_val_formatted.jsonl"),
        ("VQAv2-IDK-train.json", "VQAv2-IDK-train_formatted.jsonl"),
        ("VQAv2-IDK-val.json", "VQAv2-IDK-val_formatted.jsonl")
    ]
    
    # 转换每个文件
    for input_filename, output_filename in files_to_convert:
        input_file = os.path.join(input_dir, input_filename)
        output_file = os.path.join(output_dir, output_filename)
        
        if os.path.exists(input_file):
            convert_idk_data(input_file, output_file, coco_base_path)
        else:
            print(f"警告：文件 {input_file} 不存在，跳过")
    
    print("所有文件转换完成！")

if __name__ == "__main__":
    main()
