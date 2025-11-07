#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行IDK评估脚本

使用lmms-eval框架运行IDK评估任务
"""

import subprocess
import argparse
import os
import sys
from datetime import datetime


def run_lmms_eval(model_name, tasks, batch_size=1, limit=None, output_dir=None):
    """运行lmms-eval评估"""
    
    # 构建命令
    cmd = [
        "python", "-m", "lmms_eval",
        "--model", model_name,
        "--tasks", ",".join(tasks),
        "--batch_size", str(batch_size),
        "--log_samples"
    ]
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    if output_dir:
        cmd.extend(["--output_path", output_dir])
    
    print(f"运行命令: {' '.join(cmd)}")
    print("-" * 60)
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = "/data/dlf/code/Field-Fidelity/tools/lmms-eval:" + env.get("PYTHONPATH", "")
    
    # 运行评估
    try:
        result = subprocess.run(cmd, cwd="/data/dlf/code/Field-Fidelity/tools/lmms-eval", 
                              env=env, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"命令执行失败，返回码: {result.returncode}")
        else:
            print("评估完成！")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"运行评估时出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='IDK评估脚本')
    
    # 模型参数
    parser.add_argument('--model', type=str, required=True,
                       help='模型名称 (如: llava_hf, qwen2_vl_hf 等)')
    parser.add_argument('--model_args', type=str, default="",
                       help='模型参数 (如: pretrained=path/to/model)')
    
    # 任务参数
    parser.add_argument('--tasks', type=str, nargs='+', 
                       default=['idk_vqav2_val'],
                       help='要运行的任务列表')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批处理大小')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制评估样本数量（用于测试）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, 
                       default='/data/dlf/code/Field-Fidelity/outputs/idk_evaluation',
                       help='输出目录')
    
    # 测试参数
    parser.add_argument('--test_mode', action='store_true',
                       help='测试模式，只运行少量样本')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 测试模式设置
    if args.test_mode:
        args.limit = 10
        print("测试模式：只评估10个样本")
    
    # 构建完整的模型名称
    model_name = args.model
    if args.model_args:
        model_name += f",{args.model_args}"
    
    # 生成时间戳输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"{args.model}_{timestamp}")
    
    print(f"IDK评估开始")
    print(f"模型: {model_name}")
    print(f"任务: {args.tasks}")
    print(f"输出路径: {output_path}")
    print("=" * 60)
    
    # 运行评估
    success = run_lmms_eval(
        model_name=model_name,
        tasks=args.tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        output_dir=output_path
    )
    
    if success:
        print(f"\n✅ IDK评估完成！结果保存在: {output_path}")
    else:
        print(f"\n❌ IDK评估失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
