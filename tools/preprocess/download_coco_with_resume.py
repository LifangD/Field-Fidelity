#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COCO数据集下载脚本 - 支持断点续传

下载COCO val2014数据集并解压到指定目录
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse


def download_with_resume(url, local_path, chunk_size=8192):
    """
    支持断点续传的下载函数
    
    Args:
        url: 下载链接
        local_path: 本地保存路径
        chunk_size: 下载块大小
    """
    local_path = Path(local_path)
    
    # 检查是否已经存在部分下载的文件
    resume_byte_pos = 0
    if local_path.exists():
        resume_byte_pos = local_path.stat().st_size
        print(f"发现已下载文件，从 {resume_byte_pos / 1024 / 1024:.1f} MB 处继续下载")
    
    # 设置请求头支持断点续传
    headers = {}
    if resume_byte_pos > 0:
        headers['Range'] = f'bytes={resume_byte_pos}-'
    
    try:
        # 发送请求
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        # 检查服务器是否支持断点续传
        if resume_byte_pos > 0 and response.status_code != 206:
            print("服务器不支持断点续传，重新开始下载")
            resume_byte_pos = 0
            response = requests.get(url, stream=True, timeout=30)
        
        response.raise_for_status()
        
        # 获取文件总大小
        if 'content-length' in response.headers:
            total_size = int(response.headers['content-length'])
            if resume_byte_pos > 0:
                total_size += resume_byte_pos
        else:
            total_size = None
        
        # 创建目录
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 下载文件
        mode = 'ab' if resume_byte_pos > 0 else 'wb'
        with open(local_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=resume_byte_pos,
                unit='B',
                unit_scale=True,
                desc=local_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ 下载完成: {local_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n下载被中断，下次运行时将从断点继续")
        return False
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """
    解压ZIP文件
    
    Args:
        zip_path: ZIP文件路径
        extract_to: 解压目标目录
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    if not zip_path.exists():
        print(f"ZIP文件不存在: {zip_path}")
        return False
    
    print(f"正在解压 {zip_path.name} 到 {extract_to}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取ZIP文件中的文件列表
            file_list = zip_ref.namelist()
            
            # 使用tqdm显示解压进度
            with tqdm(total=len(file_list), desc="解压文件") as pbar:
                for file_name in file_list:
                    zip_ref.extract(file_name, extract_to)
                    pbar.update(1)
        
        print(f"✓ 解压完成: {extract_to}")
        return True
        
    except zipfile.BadZipFile:
        print(f"❌ ZIP文件损坏: {zip_path}")
        return False
    except Exception as e:
        print(f"解压失败: {e}")
        return False


def verify_download(file_path, expected_size=None):
    """
    验证下载的文件
    
    Args:
        file_path: 文件路径
        expected_size: 期望的文件大小（字节）
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return False
    
    file_size = file_path.stat().st_size
    print(f"文件大小: {file_size / 1024 / 1024:.1f} MB")
    
    if expected_size and abs(file_size - expected_size) > 1024:  # 允许1KB的误差
        print(f"❌ 文件大小不匹配，期望: {expected_size / 1024 / 1024:.1f} MB")
        return False
    
    # 尝试打开ZIP文件验证完整性
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # 测试ZIP文件完整性
            bad_file = zip_ref.testzip()
            if bad_file:
                print(f"❌ ZIP文件损坏，损坏的文件: {bad_file}")
                return False
            else:
                print("✓ ZIP文件完整性验证通过")
                return True
    except zipfile.BadZipFile:
        print("❌ ZIP文件格式错误")
        return False


def download_and_extract_dataset(dataset_name, url, output_dir, keep_zip=False, only_download=False, only_extract=False):
    """
    下载并解压单个数据集
    
    Args:
        dataset_name: 数据集名称 (如 'val2014')
        url: 下载链接
        output_dir: 输出目录
        keep_zip: 是否保留ZIP文件
        only_download: 只下载不解压
        only_extract: 只解压已下载的文件
    
    Returns:
        bool: 是否成功
    """
    output_dir = Path(output_dir)
    zip_path = output_dir / f"{dataset_name}.zip"
    
    print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")
    print(f"ZIP文件: {zip_path}")
    
    # 下载文件
    if not only_extract:
        print(f"开始下载 {dataset_name}.zip...")
        
        success = download_with_resume(url, zip_path)
        
        if not success:
            print(f"❌ {dataset_name} 下载失败")
            return False
        
        # 验证下载的文件
        print(f"验证 {dataset_name} 文件...")
        if not verify_download(zip_path):
            print(f"❌ {dataset_name} 文件验证失败")
            return False
    
    # 解压文件
    if not only_download:
        if zip_path.exists():
            print(f"开始解压 {dataset_name}...")
            success = extract_zip(zip_path, output_dir)
            
            if not success:
                print(f"❌ {dataset_name} 解压失败")
                return False
            
            # 验证解压结果
            dataset_dir = output_dir / dataset_name
            if dataset_dir.exists():
                image_count = len(list(dataset_dir.glob("*.jpg")))
                print(f"✓ {dataset_name} 解压完成，共 {image_count} 张图片")
            else:
                print(f"❌ 解压后未找到 {dataset_name} 目录")
                return False
            
            # 删除ZIP文件（如果不保留）
            if not keep_zip:
                print(f"删除ZIP文件: {zip_path}")
                zip_path.unlink()
        else:
            print(f"❌ ZIP文件不存在: {zip_path}")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='下载COCO数据集 (train2014, val2014, test2014)')
    parser.add_argument('--output-dir', type=str, 
                       default='/data/dlf/code/Field-Fidelity/data/coco',
                       help='输出目录')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['val2014', 'train2014','test2014'],
                       choices=['train2014', 'val2014', 'test2014'],
                       help='要下载的数据集 (默认: val2014 train2014)')
    parser.add_argument('--keep-zip', action='store_true',
                       help='保留ZIP文件')
    parser.add_argument('--only-download', action='store_true',
                       help='只下载不解压')
    parser.add_argument('--only-extract', action='store_true',
                       help='只解压已下载的文件')
    
    args = parser.parse_args()
    
    # COCO数据集下载链接
    dataset_urls = {
        'train2014': 'http://images.cocodataset.org/zips/train2014.zip',
        'val2014': 'http://images.cocodataset.org/zips/val2014.zip',
        'test2014': 'http://images.cocodataset.org/zips/test2014.zip'
    }
    
    # 数据集大小信息 (大约值，用于显示)
    dataset_sizes = {
        'train2014': '13.0 GB (约82,783张图片)',
        'val2014': '6.2 GB (约40,504张图片)', 
        'test2014': '6.6 GB (约40,775张图片)'
    }
    
    output_dir = Path(args.output_dir)
    
    print("COCO 数据集下载工具")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"要下载的数据集: {', '.join(args.datasets)}")
    
    # 显示数据集大小信息
    print("\n数据集大小信息:")
    for dataset in args.datasets:
        print(f"  {dataset}: {dataset_sizes[dataset]}")
    
    total_size_gb = sum([13.0 if d == 'train2014' else 6.2 if d == 'val2014' else 6.6 for d in args.datasets])
    print(f"\n预计总下载大小: ~{total_size_gb:.1f} GB")
    
    # 确认继续
    if not args.only_extract:
        response = input("\n是否继续下载? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("下载已取消")
            return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载和解压每个数据集
    success_count = 0
    failed_datasets = []
    
    for dataset_name in args.datasets:
        url = dataset_urls[dataset_name]
        
        try:
            success = download_and_extract_dataset(
                dataset_name=dataset_name,
                url=url,
                output_dir=output_dir,
                keep_zip=args.keep_zip,
                only_download=args.only_download,
                only_extract=args.only_extract
            )
            
            if success:
                success_count += 1
            else:
                failed_datasets.append(dataset_name)
                
        except KeyboardInterrupt:
            print(f"\n下载被中断，{dataset_name} 未完成")
            print("下次运行时将从断点继续")
            break
        except Exception as e:
            print(f"❌ {dataset_name} 处理失败: {e}")
            failed_datasets.append(dataset_name)
    
    # 总结
    print(f"\n{'='*60}")
    print(f"下载完成总结:")
    print(f"成功: {success_count}/{len(args.datasets)} 个数据集")
    
    if failed_datasets:
        print(f"失败: {', '.join(failed_datasets)}")
    
    # 验证最终结果
    if success_count > 0:
        print(f"\n验证下载结果...")
        for dataset_name in args.datasets:
            if dataset_name not in failed_datasets:
                dataset_dir = output_dir / dataset_name
                if dataset_dir.exists():
                    image_count = len(list(dataset_dir.glob("*.jpg")))
                    print(f"✓ {dataset_name}: {image_count} 张图片")
        
        # 检查IDK数据集需要的特定图片
        test_images = [
            "val2014/COCO_val2014_000000262162.jpg",
            "val2014/COCO_val2014_000000131108.jpg"
        ]
        
        print(f"\n检查IDK数据集需要的测试图片:")
        for img_path in test_images:
            full_path = output_dir / img_path
            if full_path.exists():
                print(f"✓ {img_path}")
            else:
                print(f"❌ {img_path}")
        
        print(f"\n✅ COCO数据集准备完成!")
        print(f"数据目录: {output_dir}")
        
        # 设置环境变量提示
        print(f"\n💡 建议设置环境变量:")
        print(f"export COCO_ROOT=\"{output_dir}\"")
        
    else:
        print(f"\n❌ 所有数据集下载失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
