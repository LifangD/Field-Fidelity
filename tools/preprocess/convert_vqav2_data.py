#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import Counter

def load_vqav2_data(vqav2_dir, split="train"):
    """加载VQAv2的问题和标注数据"""
    print(f"正在加载VQAv2 {split}数据...")
    
    # 加载问题数据
    if split == "train":
        questions_file = vqav2_dir / "questions" / "train" / "v2_OpenEnded_mscoco_train2014_questions.json"
        annotations_file = vqav2_dir / "annotations" / "train" / "v2_mscoco_train2014_annotations.json"
        image_dir_name = "train2014"
    elif split == "val":
        questions_file = vqav2_dir / "questions" / "val" / "v2_OpenEnded_mscoco_val2014_questions.json"
        annotations_file = vqav2_dir / "annotations" / "val" / "v2_mscoco_val2014_annotations.json"
        image_dir_name = "val2014"
    else:
        raise ValueError(f"不支持的split: {split}")
    
    # 加载问题
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    # 加载标注
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
    
    # 创建问题ID到标注的映射
    annotations = {}
    for ann in annotations_data['annotations']:
        annotations[ann['question_id']] = ann
    
    print(f"加载了 {len(questions_data['questions'])} 个问题和 {len(annotations)} 个标注")
    
    return questions_data, annotations, image_dir_name

def convert_vqav2_to_messages(question, answers, multiple_choice_answer=None):
    """将VQAv2格式转换为messages格式"""
    messages = []
    
    # 用户问题
    messages.append({
        "role": "user",
        "content": f"<image>{question}"
    })
    
    # 处理答案 - 优先使用multiple_choice_answer，否则使用最频繁的答案
    if multiple_choice_answer:
        assistant_answer = multiple_choice_answer
    else:
        # 统计答案频率
        answer_texts = [ans['answer'] for ans in answers]
        answer_counter = Counter(answer_texts)
        most_common_answer = answer_counter.most_common(1)[0][0]
        assistant_answer = most_common_answer
    
    messages.append({
        "role": "assistant",
        "content": assistant_answer
    })
    
    return messages

def process_vqav2_split(vqav2_dir, coco_dir, output_dir, split="train", max_examples=None):
    """处理VQAv2的一个split"""
    print(f"\n=== 处理VQAv2 {split}数据 ===")
    
    # 加载数据
    questions_data, annotations, image_dir_name = load_vqav2_data(vqav2_dir, split)
    
    # 图片目录路径
    images_dir = Path(coco_dir) / image_dir_name
    
    converted_data = []
    skipped_count = 0
    
    questions = questions_data['questions']
    if max_examples:
        questions = questions[:max_examples]
        print(f"限制处理前 {max_examples} 个样本")
    
    for i, q in enumerate(questions):
        if i % 10000 == 0:
            print(f"已处理 {i}/{len(questions)} 个问题...")
        
        qid = q['question_id']
        image_id = q['image_id']
        
        # 获取对应的标注
        ann = annotations.get(qid)
        if not ann:
            skipped_count += 1
            continue
        
        # 构建图片路径
        image_file = f"COCO_{image_dir_name}_{image_id:012d}.jpg"
        image_path = images_dir / image_file
        
        # 检查图片是否存在
        if not image_path.exists():
            skipped_count += 1
            continue
        
        # 转换为messages格式
        messages = convert_vqav2_to_messages(
            q['question'], 
            ann['answers'], 
            ann.get('multiple_choice_answer')
        )
        
        # 构建训练样本
        solution = messages[-1]['content']
        del messages[-1]
        example = {
            "messages": messages,
            "images": str(image_path),
            "solution": solution  # 将最后一条assistant回答放到solution字段
        }
        
        converted_data.append(example)
    
    print(f"处理完成: {len(converted_data)} 个有效样本, {skipped_count} 个跳过")
    
    # 保存转换后的数据
    output_file = Path(output_dir) / f"vqav2_{split}_formatted.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in converted_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"数据已保存到: {output_file}")
    return len(converted_data)

def main():
    """主函数"""
    # 设置路径
    base_dir = Path("/data/dlf/code/Field-Fidelity/data")
    vqav2_dir = base_dir / "vqav2"
    coco_dir = base_dir / "coco"
    output_dir = base_dir / "vqav2" / "formatted"
    
    print("开始转换VQAv2数据为训练格式...")
    print(f"VQAv2数据目录: {vqav2_dir}")
    print(f"COCO图片目录: {coco_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查必要的目录是否存在
    if not vqav2_dir.exists():
        print(f"❌ VQAv2数据目录不存在: {vqav2_dir}")
        return
    
    if not coco_dir.exists():
        print(f"❌ COCO图片目录不存在: {coco_dir}")
        return
    
    total_examples = 0
    
    # 处理训练集
    try:
        count = process_vqav2_split(vqav2_dir, coco_dir, output_dir, "train")
        total_examples += count
    except Exception as e:
        print(f"❌ 处理训练集失败: {e}")
    
    # 处理验证集
    try:
        count = process_vqav2_split(vqav2_dir, coco_dir, output_dir, "val")
        total_examples += count
    except Exception as e:
        print(f"❌ 处理验证集失败: {e}")
    
    print(f"\n=== 转换完成 ===")
    print(f"总共转换了 {total_examples} 个训练样本")
    print(f"数据保存在: {output_dir}")
    
    # 显示输出文件信息
    if output_dir.exists():
        print("\n生成的文件:")
        for file in output_dir.glob("*.jsonl"):
            file_size = file.stat().st_size / (1024 * 1024)  # MB
            with open(file, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"  {file.name}: {line_count} 行, {file_size:.1f} MB")

if __name__ == "__main__":
    main()
