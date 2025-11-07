#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPT-4o 中国画标注处理器
参考 SpecificQA/img_err_detect.py 的架构实现
"""

import os
import argparse
import json
import base64
import logging
from typing import Dict, Any, List
from copy import deepcopy
from tqdm import tqdm
import uuid
from openai import OpenAI
from PIL import Image
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
# 设置日志
import requests
from transformers import AutoTokenizer
def setup_logging(level: str = 'INFO'):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    if not os.path.exists(file_path):
        return []
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logging.warning(f"跳过无效的JSON行: {e}")
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """保存数据到JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_prompt_template(template_path: str) -> str:
    """加载提示模板"""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def resize_image_if_needed(image: Image.Image, max_dimension: int = 16000) -> Image.Image:
    """如果图片超过最大尺寸则调整大小"""
    width, height = image.size
    original_size = width * height
    
    if width > max_dimension or height > max_dimension:
        aspect_ratio = width / height
        if width > height:
            new_width = max_dimension
            new_height = int(max_dimension / aspect_ratio)
        else:
            new_height = max_dimension
            new_width = int(max_dimension * aspect_ratio)
        
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        new_size = new_width * new_height
        compression_ratio = new_size / original_size
        
        logging.info(f"图片已缩放: {width}x{height} -> {new_width}x{new_height} (压缩比: {compression_ratio:.3f})")
        return resized_image
    
    return image

def format_meta_info(meta_data: Dict[str, Any]) -> str:
    """格式化元数据信息"""
    if isinstance(meta_data, dict):
        # 移除annotation字段（如果存在）
        cleaned_meta = {k: v for k, v in meta_data.items() if k != 'annotation'}
        return json.dumps(cleaned_meta, ensure_ascii=False, indent=2)
    return str(meta_data)

def painting_prompt_template_filler(template: str, item: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """中国画特定的模板填充函数
    
    Args:
        template: 提示模板
        item: 数据项
        **kwargs: 其他参数
        
    Returns:
        包含prompt和其他信息的字典
    """
    # 提取基本信息
    # meta_info = eval(item.get("meta", {}))
    
    # if meta_info.get("work").get("create_year", "") == meta_info.get("author", {}).get("birth_year", ""):
    #     del meta_info["work"]["create_year"]
    
    # # 格式化元数据
    # json_text = format_meta_info(meta_info)
    question = item.get("question", "")
    current_answer = item.get("current_answer", "")
    if "<answer>" in current_answer and "</answer>" in current_answer:
        matches = re.findall("<answer>(.*?)</answer>", current_answer, re.DOTALL)
        if len(matches) > 0:
            current_answer = matches[0]
      
    reference_answer = item.get("reference_answer", "")
    
    # 填充模板
    filled_prompt = template.replace("{question}", question).replace("{current_answer}", current_answer).replace("{reference_answer}", reference_answer)
    
    # 返回结果
    #print("check_filled_prompt:\n", filled_prompt)
    if "original_images" in item:
        img_path = item.get("original_images", [])[0].get("path", "")
    else:
        img_path = item.get("original_input", {}).get("images", [])[0].get("path", "")
    result = {
        "prompt": filled_prompt,
        "original_data": item,
        "img_path": img_path,
        "current_answer": current_answer
    }
    
    return result

def generate_prompts(data: List[Dict[str, Any]], 
                    template: str, 
                    template_filler=painting_prompt_template_filler,
                    use_image_path: bool = True,
                    image_path_key: str = "img") -> List[Dict[str, Any]]:
    """生成提示列表"""
    prompts = []
    for item in data:
        prompt_info = template_filler(template, item)
        
        if use_image_path and image_path_key in item:
            prompt_info["img_path"] = item[image_path_key]
        
        prompts.append(prompt_info)
    
    return prompts

class GPT4OClient:
    """GPT-4o API客户端"""
    
    def __init__(self, base_url: str, api_key: str, max_workers: int, model_name: str):
        self.base_url = base_url
        self.api_key = api_key
        self.max_workers = max_workers
        # 使用线程本地存储，每个线程使用独立的客户端
        self.thread_local = threading.local()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_client(self):
        """获取线程本地的OpenAI客户端"""
        if not hasattr(self.thread_local, 'client'):
            self.thread_local.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self.thread_local.client
    
    def construct_messages(self, prompt: str, image_base64: str | None = None, response: str | None = None) -> List[Dict[str, Any]]:
        """构造消息"""
        content = [{"type": "text", "text": prompt}]
        
        if image_base64:
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })  # type: ignore
       
        
        return [{"role": "user", "content": content},  {"role": "assistant", "content": response}]
    
    def process_convs(self,conv, base_url, tokenizer, model_name_or_path):
        payload = {"model": model_name_or_path}
        convs_formatted = []
      
        conv = tokenizer.apply_chat_template(conv, tokenize=False)
        if tokenizer.bos_token is not None and conv.startswith(tokenizer.bos_token):
            conv = conv[len(tokenizer.bos_token) :]
        convs_formatted.append(conv)

        payload.update({"text": convs_formatted})
       
        responses = requests.post(base_url, json=payload).json()
        
    
        return  responses[0]["embedding"][0]

   
    
    def process_single_item(self, item: Dict[str, Any], temperature: float = 0.2, 
                           top_p: float = 0.1, max_tokens: int = 8192, 
                           max_image_dimension: int = 16000, model_name: str = "openai/gpt_4o",
                           use_img: bool = True) -> Dict[str, Any] | None:
        """处理单个项目"""
 
        # prompt = item["prompt"]
        # img_path = item.get("img_path", "") if use_img else ""
        # original_data = item["original_data"]
    
        # response = self.generate_single(
        #     prompt=prompt,
        #     image_path=img_path,
        #     temperature=temperature,
        #     top_p=top_p,
        #     max_tokens=max_tokens,
        #     max_image_dimension=max_image_dimension,
        #     model_name=model_name,
        #     response=item["current_answer"]
        # )
        
        
        # if response:
        #     result = create_output_data(original_data, response, prompt, model_name)
        #     return result
        # else:
        #     logging.warning(f"生成的响应为空: {item.get('img_path', 'unknown')}")
        #     return None

        convs = [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["current_answer"]}]
        result = self.process_convs(convs, self.base_url, self.tokenizer, model_name)
        
        if result:
            result = create_output_data(item["original_data"], str(result), item["prompt"], model_name)
            return result
        else:
            logging.warning(f"生成的响应为空: {item.get('img_path', 'unknown')}")
            return None

    
    def process_batch_multithread(self, batch: List[Dict[str, Any]], 
                                 temperature: float = 0.2, top_p: float = 0.1,
                                 max_tokens: int = 8192, max_image_dimension: int = 16000,
                                 model_name: str = "openai/gpt_4o", use_img: bool = True) -> List[Dict[str, Any]]:
        """多线程处理批次"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(
                    self.process_single_item, 
                    item, temperature, top_p, max_tokens, 
                    max_image_dimension, model_name, use_img
                ): item for item in batch
            }
            
            # 收集结果
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logging.error(f"线程处理失败 {item.get('img_path', 'unknown')}: {e}")
        
        return results

def create_output_data(original_data: Dict[str, Any], 
                      response: str, 
                      prompt: str,
                      model_name: str = "gpt-4o") -> Dict[str, Any]:
    """创建输出数据结构"""
    result = original_data.copy()
    new_description = response
    if "--待审核--" in response:
        new_description = new_description.replace("--待审核--","").strip()
    result.update({
        # "id_": original_data.get("raw_image_path").split("/")[-1].split(".")[0],
        # "new_prompt": prompt,
        # "new_description": new_description,
        "model": model_name,
        "raw_model_output": response
    })
    return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GPT-4o 中国画标注处理器')
    parser.add_argument('--data', type=str, required=True, help='输入数据路径')
    parser.add_argument('--template', type=str, required=True, help='提示模板路径')
    parser.add_argument('--output', type=str, required=True, help='输出结果路径')
    parser.add_argument('--base_url', type=str, 
                       default='https://api.zjuqx.cn/v1', 
                       help='API服务URL')
    parser.add_argument('--api_key', type=str, 
                       #default='sk-jQMhSRpAFMH9p3SaA255Fd5fB1Fe4d348a9d79Dc223c423b',
                       default='sk-DyIxTyw8hEOt8Wxo75D64989687044E28cE3Db73F735C09f',
                       help='API密钥')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小（GPT-4o建议使用1）')
    parser.add_argument('--sample_size', type=int, default=0, help='采样大小，0表示使用全部数据')
    parser.add_argument('--log_level', type=str, default='INFO', help='日志级别')
    parser.add_argument('--use_img', action='store_true', help='是否使用图像')
    parser.add_argument('--temperature', type=float, default=0.6, help='温度参数')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p参数')
    parser.add_argument('--max_image_dimension', type=int, default=16000, help='图片最大尺寸，超过此尺寸将被缩放')
    parser.add_argument('--model_name',type=str,default="/data/share/hub/models/Qwen/Qwen3-VL-235B-A22B-Instruct")
    parser.add_argument('--max_workers', type=int, default=8, help='多线程处理的最大线程数')
    parser.add_argument('--use_multithread', action='store_true', help='是否使用多线程处理')

    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 记录配置信息
    logging.info("=== GPT-4o 处理器启动 ===")
    logging.info(f"输入数据: {args.data}")
    logging.info(f"输出路径: {args.output}")
    logging.info(f"使用图像: {args.use_img}")
    logging.info(f"最大图片尺寸: {args.max_image_dimension}")
    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"采样大小: {args.sample_size if args.sample_size > 0 else '全部'}")
    logging.info(f"温度参数: {args.temperature}")
    logging.info(f"Top-p参数: {args.top_p}")
    logging.info(f"多线程处理: {args.use_multithread}")
    logging.info(f"最大线程数: {args.max_workers}")
    
    # 加载数据
    data = load_jsonl(args.data)
    logging.info(f"加载了 {len(data)} 条数据")
    
    # 恢复逻辑 - 检查已处理的数据
    processed_ids = set()
    if os.path.exists(args.output):
        logging.info(f"输出文件 {args.output} 已存在，将进行增量处理")
        try:
            processed_data = load_jsonl(args.output) 
            for item in processed_data:
                if "original_images" in item:
                    img_path = item.get("original_images", [])[0].get("path", "")
                else:
                    img_path = item.get("original_input", {}).get("images", [])[0].get("path", "")
                processed_ids.add(img_path)
            logging.info(f"找到 {len(processed_ids)} 条已处理的数据")
        except Exception as e:
            logging.warning(f"无法加载已存在的输出文件: {e}")
            processed_ids = set()
    
    # 过滤已处理的数据
    if processed_ids:
        original_count = len(data)
        
        # 创建新的列表来存储未处理的数据
        filtered_data = []
        for item in tqdm(data, desc="过滤已处理的数据"):
            if "original_images" in item:
                img_path = item.get("original_images", [])[0].get("path", "")
            else:
                img_path = item.get("original_input", {}).get("images", [])[0].get("path", "")
            
            if img_path not in processed_ids:
                filtered_data.append(item)
        
        data = filtered_data  # 替换原数据
        logging.info(f"过滤后，剩余 {len(data)} / {original_count} 条数据需要处理")
        #return
    
    # 加载提示模板
    template = load_prompt_template(args.template)
    
    # 生成提示
    prompts = generate_prompts(
        data=data,
        template=template,
        template_filler=painting_prompt_template_filler,
        use_image_path=args.use_img,
        image_path_key="raw_image_path"
    )
    
    # 初始化客户端
    client = GPT4OClient(args.base_url, args.api_key, args.max_workers,args.model_name)
    
    # 批处理
    logging.info(f"开始处理 {len(prompts)} 条数据...")
    total_new_results = 0
    if args.sample_size>0:
        prompts = prompts[:args.sample_size]
    # 根据是否使用多线程选择不同的处理方式
    if args.use_multithread:
        logging.info("使用多线程处理模式")
        # 多线程模式：批次大小建议设置为线程数的倍数
        effective_batch_size = max(args.batch_size, args.max_workers * 2)
        
        for i in tqdm(range(0, len(prompts), effective_batch_size), desc="多线程批次处理"):
            batch = prompts[i:i+effective_batch_size]
            
            start_time = time.time()
            batch_results = client.process_batch_multithread(
                batch=batch,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=8192,
                max_image_dimension=args.max_image_dimension,
                model_name=args.model_name,
                use_img=args.use_img
            )
            end_time = time.time()
            
            # 保存批次结果
            if batch_results:
                with open(args.output, 'a', encoding='utf-8') as f:
                    for item in batch_results:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                total_new_results += len(batch_results)
                
                processing_time = end_time - start_time
                items_per_second = len(batch_results) / processing_time if processing_time > 0 else 0
                logging.info(f"多线程批次处理完成，保存 {len(batch_results)}/{len(batch)} 条结果，"
                           f"耗时 {processing_time:.2f}秒，速度 {items_per_second:.2f} 项/秒")
    else:
        logging.info("使用单线程处理模式")
        # 原有的单线程处理逻辑
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="单线程批次处理"):
            batch = prompts[i:i+args.batch_size]
            
            batch_results = []
            for item in batch:
                prompt = item["prompt"]
                img_path = item.get("img_path", "") if args.use_img else ""
                original_data = item["original_data"]
                
                logging.info(f"处理: {item.get('img_path', 'unknown')}")
                
                try:
                    response = client.generate_single(
                        prompt=prompt,
                        image_path=img_path,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=8192,
                        max_image_dimension=args.max_image_dimension,
                        model_name=args.model_name
                    )
                    
                    if response:
                        result = create_output_data(original_data, response, prompt, args.model_name)
                        batch_results.append(result)
                        logging.info(f"成功生成响应，长度: {len(response)}")
                    else:
                        logging.warning("生成的响应为空")
                        
                except Exception as e:
                    logging.error(f"处理失败: {e}")
                    continue
            
            # 保存批次结果
            if batch_results:
                with open(args.output, 'a', encoding='utf-8') as f:
                    for item in batch_results:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                total_new_results += len(batch_results)
                logging.info(f"单线程批次处理完成，保存 {len(batch_results)} 条结果")
    
    print(f"处理完成，共生成 {total_new_results} 条新结果，已保存到 {args.output}")

if __name__ == '__main__':
    main()



# 单线程模式:
# python /data/dlf/code/Field-Fidelity/tools/filter_rm_reasoing/get_single_reward.py \
#   --data /data/dlf/code/Field-Fidelity/outputs/experiments/grpo/logs/reward_evaluation_20251024_095439/reward_details.jsonl \
#   --template /data/dlf/code/Field-Fidelity/src/train/prompt/reward_process_atomic.md \
#   --output /data/dlf/code/Field-Fidelity/outputs/results/4o_rm_20251024_095439_atomic_vl.jsonl \
#   --sample_size 30000 --batch_size 16 --base_url http://10.160.199.226:8001/v1
#
# 多线程模式（推荐）:
'''
python /data/dlf/code/Field-Fidelity/tools/filter_rm_reasoing/get_sky_rm.py \
  --data /data/dlf/code/Field-Fidelity/outputs/experiments/grpo/logs/reward_evaluation_20251024_095439/reward_details.jsonl \
  --template /data/dlf/code/Field-Fidelity/src/train/prompt/sky_rm.md \
  --output /data/dlf/code/Field-Fidelity/outputs/results/4o_rm_20251024_095439_sky_rm.jsonl \
  --sample_size 32 --batch_size 8 --use_multithread --max_workers 2 \
  --model_name /data/share/hub/models/Skywork/Skywork-Reward-V2-Llama-3___1-8B-40M \
  --base_url http://127.0.0.1:8003/classify
'''