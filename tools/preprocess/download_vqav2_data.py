#!/usr/bin/env python3
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import json

def download_file(url, local_path, description=""):
    """下载文件并显示进度条"""
    print(f"正在下载 {description}: {url}")
    
    # 创建目录
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # 如果文件已存在，跳过下载
    if os.path.exists(local_path):
        print(f"文件已存在，跳过下载: {local_path}")
        return local_path
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"下载完成: {local_path}")
    return local_path

def extract_zip(zip_path, extract_to):
    """解压ZIP文件"""
    print(f"正在解压: {zip_path} -> {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"解压完成: {extract_to}")

def download_vqav2_data(base_dir="/data/dlf/code/Field-Fidelity/data"):
    """下载VQAv2数据集"""
    
    # VQAv2数据集的URL
    urls = {
        "questions": {
            # "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
            # "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
            # "test-dev": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
            "test": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip"
        },
        # "annotations": {
        #     "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        #     "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        # },
        # "images": {
        #     "train": "http://images.cocodataset.org/zips/train2014.zip",
        #     "val": "http://images.cocodataset.org/zips/val2014.zip",
        #     "test-dev": "http://images.cocodataset.org/zips/test2015.zip",
        # },
    }
    
    # 文件名映射
    sub_folder_names = {
        "questions": {
            "train": "v2_OpenEnded_mscoco_train2014_questions.json",
            "val": "v2_OpenEnded_mscoco_val2014_questions.json",
            "test-dev": "v2_OpenEnded_mscoco_test-dev2015_questions.json",
        },
        "annotations": {
            "train": "v2_mscoco_train2014_annotations.json",
            "val": "v2_mscoco_val2014_annotations.json",
        },
        # "images": {
        #     "train": "train2014",
        #     "val": "val2014",
        #     "test-dev": "test2015",
        # },
    }
    
    # 创建VQAv2数据目录
    vqav2_dir = Path(base_dir) / "vqav2"
    vqav2_dir.mkdir(exist_ok=True)
    
    # 下载和解压各类数据
    for data_type, type_urls in urls.items():
        print(f"\n=== 下载 {data_type} 数据 ===")
        
        type_dir = vqav2_dir / data_type
        type_dir.mkdir(exist_ok=True)
        
        for split, url in type_urls.items():
            # 下载ZIP文件
            zip_filename = f"{data_type}_{split}.zip"
            zip_path = type_dir / zip_filename
            
            try:
                download_file(url, zip_path, f"{data_type} {split}")
                
                # 解压文件
                extract_dir = type_dir / split
                extract_zip(zip_path, extract_dir)
                
                # 检查解压后的文件
                if data_type in ["questions", "annotations"]:
                    expected_file = extract_dir / sub_folder_names[data_type][split]
                    if expected_file.exists():
                        print(f"✓ 找到预期文件: {expected_file}")
                    else:
                        print(f"⚠ 未找到预期文件: {expected_file}")
                        # 列出实际解压的文件
                        print("实际解压的文件:")
                        for file in extract_dir.rglob("*"):
                            if file.is_file():
                                print(f"  {file}")
                
                elif data_type == "images":
                    expected_dir = extract_dir / sub_folder_names[data_type][split]
                    if expected_dir.exists():
                        image_count = len(list(expected_dir.glob("*.jpg")))
                        print(f"✓ 找到图片目录: {expected_dir} (包含 {image_count} 张图片)")
                    else:
                        print(f"⚠ 未找到预期目录: {expected_dir}")
                
                # 删除ZIP文件以节省空间
                if zip_path.exists():
                    os.remove(zip_path)
                    print(f"已删除ZIP文件: {zip_path}")
                    
            except Exception as e:
                print(f"❌ 下载 {data_type} {split} 失败: {e}")
                continue
    
    print(f"\n=== VQAv2数据下载完成 ===")
    print(f"数据保存在: {vqav2_dir}")
    
    # 显示目录结构
    print("\n目录结构:")
    for item in sorted(vqav2_dir.rglob("*")):
        if item.is_file() and item.suffix in ['.json']:
            rel_path = item.relative_to(vqav2_dir)
            file_size = item.stat().st_size / (1024 * 1024)  # MB
            print(f"  {rel_path} ({file_size:.1f} MB)")
        elif item.is_dir() and any(item.glob("*.jpg")):
            rel_path = item.relative_to(vqav2_dir)
            image_count = len(list(item.glob("*.jpg")))
            print(f"  {rel_path}/ ({image_count} 张图片)")

def main():
    """主函数"""
    print("开始下载VQAv2数据集...")
    download_vqav2_data()

if __name__ == "__main__":
    main()
