#!/usr/bin/env python3
import json
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import io

def load_rlhf_data(rlhf_file):
    """加载RLHF-V-Dataset数据"""
    print(f"正在加载RLHF-V-Dataset数据: {rlhf_file}")
    
    df = pd.read_parquet(rlhf_file)
    print(f"加载了 {len(df)} 个样本")
    
    return df

def convert_rlhf_to_messages(question, chosen_answer):
    """将RLHF格式转换为messages格式"""
    messages = []
    
    # 用户问题
    messages.append({
        "role": "user", 
        "content": f"<image>{question}"
    })
    
    # 选择的答案作为assistant回答
    messages.append({
        "role": "assistant",
        "content": chosen_answer
    })
    
    return messages

def save_image_from_data(image_data, output_path):
    """从图片数据保存图片"""
    try:
        # 检查image_data的类型
        if isinstance(image_data, dict) and 'bytes' in image_data:
            # 从字典中获取二进制数据
            binary_data = image_data['bytes']
        else:
            # 如果是直接的二进制数据
            binary_data = image_data
            
        image = Image.open(io.BytesIO(binary_data))
        
        # 如果图片是调色板模式(P)或者有透明通道(RGBA)，转换为RGB
        if image.mode in ('P', 'RGBA', 'LA'):
            # 如果有透明通道，先转换为RGBA再转RGB
            if image.mode == 'P' and 'transparency' in image.info:
                image = image.convert('RGBA')
            if image.mode in ('RGBA', 'LA'):
                # 创建白色背景
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'LA':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1])  # 使用alpha通道作为mask
                image = background
            else:
                image = image.convert('RGB')
        
        # 确保输出目录存在
        os.makedirs(output_path.parent, exist_ok=True)
        image.save(output_path, 'JPEG')
        return True
    except Exception as e:
        print(f"保存图片失败: {e}")
        return False

def process_rlhf_data(rlhf_file, output_dir, images_dir, max_examples=None):
    """处理RLHF-V-Dataset数据"""
    print(f"\n=== 处理RLHF-V-Dataset数据 ===")
    
    # 加载数据
    df = load_rlhf_data(rlhf_file)
    
    converted_data = []
    skipped_count = 0
    
    # 限制处理数量
    if max_examples:
        df = df.head(max_examples)
        print(f"限制处理前 {max_examples} 个样本")
    
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"已处理 {i}/{len(df)} 个样本...")
        
        try:
            # 解析text字段中的JSON数据
            text_data = json.loads(row['text'])
            question = text_data['question']
            chosen_answer = text_data['chosen']
            rejected_answer = text_data['rejected']
            
            # 构建图片保存路径
            image_filename = f"rlhf_{row['idx']:06d}.jpg"
            image_path = Path(images_dir) / image_filename
            
            # 保存图片
            if not save_image_from_data(row['image'], image_path):
                skipped_count += 1
                continue
            
            # 转换为messages格式
            messages = convert_rlhf_to_messages(question, chosen_answer)
            
            # 构建训练样本
            solution = messages[-1]['content']
            del messages[-1]
            example = {
                "messages": messages,
                "images": str(image_path),
                "solution": solution,  # 将最后一条assistant回答放到solution字段
                "rejected_answer": rejected_answer
            }
            
            converted_data.append(example)
            
        except Exception as e:
            print(f"处理第{i}行数据时出错: {e}")
            skipped_count += 1
            continue
    
    print(f"处理完成: {len(converted_data)} 个有效样本, {skipped_count} 个跳过")
    
    # 保存转换后的数据
    output_file = Path(output_dir) / "rlhf_formatted.jsonl"
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
    rlhf_file = Path("/data/dlf/code/Muffin/RLHF-V-Dataset/RLHF-V-Dataset.parquet")
    output_dir = base_dir / "rlhf" / "formatted"
    images_dir = base_dir / "rlhf" / "images"
    
    print("开始转换RLHF-V-Dataset数据为训练格式...")
    print(f"RLHF数据文件: {rlhf_file}")
    print(f"输出目录: {output_dir}")
    print(f"图片目录: {images_dir}")
    
    # 检查必要的文件是否存在
    if not rlhf_file.exists():
        print(f"❌ RLHF数据文件不存在: {rlhf_file}")
        return
    
    total_examples = 0
    
    # 处理数据
    try:
        count = process_rlhf_data(rlhf_file, output_dir, images_dir)
        total_examples += count
    except Exception as e:
        print(f"❌ 处理数据失败: {e}")
    
    print(f"\n=== 转换完成 ===")
    print(f"总共转换了 {total_examples} 个训练样本")
    print(f"数据保存在: {output_dir}")
    print(f"图片保存在: {images_dir}")
    
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
