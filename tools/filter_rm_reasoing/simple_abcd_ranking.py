#!/usr/bin/env python3
"""
简化的ABCD答案排序分析脚本
1. 将4个候选答案标记为A、B、C、D
2. 根据final_score得到偏序排序（如BACD）
3. 使用VLM基于reference_answer和图片对ABCD重新排序
4. 输出两个排序结果和原来的打分理由
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: 未安装openai库，VLM比较功能将不可用")

@dataclass
class Answer:
    """答案数据结构"""
    label: str  # A, B, C, D
    content: str
    score: float
    reasoning: str
    timestamp: str
    image_path: str

class SimpleABCDAnalyzer:
    def __init__(self, jsonl_file: str, model_url: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None, use_img: bool = True, limit: Optional[int] = None, margin: float = 0.0, max_workers: int = 4):
        self.jsonl_file = jsonl_file
        self.questions_data = defaultdict(list)
        self.use_img = use_img
        self.limit = limit
        self.margin = margin
        self.max_workers = max_workers
        self.model_url = model_url
        self.api_key = api_key
        self.model = model
        
        if OPENAI_AVAILABLE and model_url and model and api_key:
            # 创建线程本地存储，每个线程使用独立的客户端
            self.thread_local = threading.local()
            self.use_vlm = True
        else:
            self.use_vlm = False
            if not OPENAI_AVAILABLE:
                print("警告: 未安装openai库")
            elif not (model_url and model and api_key):
                print("警告: 未提供模型配置，将跳过VLM比较")
    
    def get_client(self):
        """获取线程本地的OpenAI客户端"""
        if not hasattr(self.thread_local, 'client'):
            self.thread_local.client = openai.OpenAI(
                base_url=self.model_url, 
                api_key=self.api_key
            )
        return self.thread_local.client
    
    def extract_answer_content(self, full_content: str) -> str:
        """提取答案内容（去掉思考过程）"""
        if '<answer>' in full_content and '</answer>' in full_content:
            start = full_content.find('<answer>') + 8
            end = full_content.find('</answer>')
            return full_content[start:end].strip()
        return full_content.strip()
    def extract_score(self,full_content)->float:
        score_patterns = [
                r'boxed\{([0-9]*\.?[0-9]+)\}'
                # r"最终评分：\s*([-+]?\d*\.?\d+)\s*分",
                # r"最终得分：\s*([-+]?\d*\.?\d+)\s*分",
                # r"分数：\s*([-+]?\d*\.?\d+)\s*分",
                # r"得分：\s*([-+]?\d*\.?\d+)\s*分",
                # r"评分：\s*([-+]?\d*\.?\d+)\s*分",
                # r"最终答案：\s*([-+]?\d*\.?\d+)\s*分?",
                # r"\\boxed{\s*([-+]?\d*\.?\d+)\s*}",  # LaTeX boxed格式
                # r"答案：\s*([-+]?\d*\.?\d+)\s*分?",
                # r"分数为[：:]\s*([-+]?\d*\.?\d+)\s*分?",
                # r"[（(]([-+]?\d*\.?\d+)[分)）]"  # 括号中的分数
            ]
            
        score_found = False
        for pattern in score_patterns:
            score_match = re.search(pattern, full_content, re.IGNORECASE)
            if score_match:
                try:
                    final_score = float(score_match.group(1))
                    score_found = True
                    break
                except ValueError:
                    continue
        if score_found:
            return final_score
        
        return float(full_content) if full_content else 0.0
        
    
    def load_data(self):
        """加载JSONL数据并按问题+图片分组"""
        print("正在加载数据...")
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                question = data['question']
                image_path = data.get('original_images', [{}])[0].get('path', '')
                
                # 使用question+图片路径作为key
                key = f"{question}|{image_path}"
                
                answer = Answer(
                    label="",  # 稍后分配
                    content=self.extract_answer_content(data['current_answer']),
                    score=self.extract_score(data.get('raw_model_output', '')),
                    reasoning=data.get('raw_model_output', ''),
                    timestamp=data['timestamp'],
                    image_path=image_path
                )
                
                self.questions_data[key].append({
                    'wrapped_answer': answer,
                    'reference_answer': data['reference_answer'],
                    'current_answer': data['current_answer'],
                    'question': question,
                    'image_path': image_path,
                    "full_content": data['raw_model_output']
                    
                })
        
        print(f"加载完成，共找到 {len(self.questions_data)} 个不同的问题+图片组合")
        show_cnt = 0
        for key, answers in self.questions_data.items():
            question, image_path = key.split('|', 1)
            print(f"问题: {question[:50]}... | 图片: {image_path.split('/')[-1]} - {len(answers)} 个答案")
            show_cnt += 1
            if show_cnt > 10:
                break
        print("="*100)
    
    def group_by_tiers(self, sorted_items: List[Dict]) -> List[List[Dict]]:
        """根据分数差异分档，差值小于margin的归为一档"""
        if not sorted_items or self.margin <= 0:
            return [[item] for item in sorted_items]
        
        tiers = []
        current_tier = [sorted_items[0]]
        
        for i in range(1, len(sorted_items)):
            prev_score = sorted_items[i-1]['wrapped_answer'].score
            curr_score = sorted_items[i]['wrapped_answer'].score
            
            # 如果分数差大于margin，开启新档
            if prev_score - curr_score > self.margin:
                tiers.append(current_tier)
                current_tier = [sorted_items[i]]
            else:
                current_tier.append(sorted_items[i])
        
        tiers.append(current_tier)
        return tiers
    
    def format_tiered_ranking(self, tiers: List[List[Dict]]) -> str:
        """格式化分档排序结果，如 A|BCD"""
        tier_labels = []
        for tier in tiers:
            # 同一档内按字母序排列
            tier_labels_list = sorted([item['wrapped_answer'].label for item in tier])
            tier_labels.append(''.join(tier_labels_list))
        return '|'.join(tier_labels)
    
    def assign_labels_and_sort_by_score(self, question_data: List[Dict]) -> tuple[List[Dict], str]:
        """分配A、B、C、D标签并按分数分档排序"""
        # 按时间戳排序确保一致性，然后分配标签
        question_data.sort(key=lambda x: x['wrapped_answer'].timestamp)
        labels = ['A', 'B', 'C', 'D']
        
        for i, item in enumerate(question_data):
            if i < len(labels):
                item['wrapped_answer'].label = labels[i]
        
        # 按分数排序（降序）
        score_sorted = sorted(question_data, key=lambda x: x['wrapped_answer'].score, reverse=True)
        
        # 分档
        tiers = self.group_by_tiers(score_sorted)
        score_ranking = self.format_tiered_ranking(tiers)
        
        return question_data, score_ranking
    
    def create_vlm_ranking_prompt(self, question: str, reference_answer: str, answers: List[Answer]) -> str:
        """创建VLM排序的prompt"""
        prompt = f"""请对以下4个候选答案进行质量排序。

问题: {question}
参考答案: {reference_answer}

候选答案:
"""
        for answer in answers:
            prompt += f"\n{answer.label}. {answer.content}"
        
        margin_desc = f"（质量差异需足够明显，差异较小时应归为同一档）" if self.margin > 0 else ""
        if self.use_img:
            prompt += f"""

请仔细观察图片，并基于以下标准对答案A、B、C、D进行分档排序{margin_desc}：
1. 与参考答案的匹配度（第一优先级）
2. 与图片内容的一致性
3. 答案的准确性和完整性
4. 不确定性表达的合理性

要求：
- 只有质量差异较大时才分到不同档位
- 质量相近的答案应归为同一档，用|分隔不同档位
- 同一档内的答案按字母序排列

输出格式：\\boxed{{A|BCD}} 表示A单独一档，B、C、D在同一档
示例：\\boxed{{A|BCD}} 或 \\boxed{{AB|CD}} 或 \\boxed{{ABCD}}

排序结果:"""
        else:
            prompt += f"""

请基于以下标准对答案A、B、C、D进行分档排序{margin_desc}：
1. 与参考答案的匹配度
2. 答案的准确性和完整性
3. 不确定性表达的合理性

要求：
- 只有质量差异较大时才分到不同档位
- 质量相近的答案应归为同一档，用|分隔不同档位
- 同一档内的答案按字母序排列

输出格式：\\boxed{{A|BCD}} 表示A单独一档，B、C、D在同一档
示例：\\boxed{{A|BCD}} 或 \\boxed{{AB|CD}} 或 \\boxed{{ABCD}}

排序结果:"""
        
        return prompt
    
    def get_vlm_ranking(self, question: str, reference_answer: str, answers: List[Answer], image_path: str) -> str:
        """使用VLM获取答案排序"""
        if not self.use_vlm or len(answers) != 4:
            return "无法排序"
        
        prompt = self.create_vlm_ranking_prompt(question, reference_answer, answers)
        try:
            # 构建消息内容
            content_items: List[Any] = [{"type": "text", "text": prompt}]
            
            # 如果使用图片，添加图片内容
            if self.use_img:
                if not os.path.exists(image_path):
                    print(f"警告: 图片文件不存在: {image_path}")
                    return "图片不存在"
                
                import base64
                with open(image_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                
                content_items.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                })
            
            messages = [{"role": "user", "content": content_items}]
            
            # 使用线程安全的客户端
            client = self.get_client()
            response = client.chat.completions.create(
                model=self.model,  # type: ignore
                messages=messages,  # type: ignore
                temperature=0.1,
                max_tokens=2048
            )
            
            vlm_response = response.choices[0].message.content or ""
            
            return vlm_response.strip()
                
        except Exception as e:
            print(f"VLM调用失败: {e}")
            return "VLM调用失败"
    
    def analyze_question(self, key: str, question_data: List[Dict]) -> Optional[Dict[str, Any]]:
        """分析单个问题的答案排序"""
        question, image_path = key.split('|', 1)
        
        if len(question_data) != 4:
            return None
        
        # 分配标签并获取分数排序
        labeled_data, score_ranking = self.assign_labels_and_sort_by_score(question_data)
        
        # 获取VLM排序
        vlm_ranking = "未获取"
        vlm_response = ""
        if self.use_vlm:
            answers = [item['wrapped_answer'] for item in labeled_data]
            image_path = answers[0].image_path
            vlm_response = self.get_vlm_ranking(
                question, 
                question_data[0]['reference_answer'], 
                answers, 
                image_path
            )

            import re
            vlm_ranking = "未获取"
            boxed_match = re.search(r'\\boxed\{(.*?)\}', vlm_response)
            if boxed_match:
                vlm_ranking = boxed_match.group(1)

        
        return {
            'question': question,
            'reference_answer': question_data[0]['reference_answer'],
            'answers': [
                {
                    'label': item['wrapped_answer'].label,
                    'content': item['wrapped_answer'].content,
                    'score': item['wrapped_answer'].score,
                    'reasoning': item['wrapped_answer'].reasoning
                }
                for item in labeled_data
            ],
            'score_ranking': score_ranking,
            'vlm_ranking': vlm_ranking,
            'vlm_response': vlm_response,
            'image_path': labeled_data[0]['wrapped_answer'].image_path
        }
    
    def process_single_question(self, item) -> Optional[Dict[str, Any]]:
        """处理单个问题（用于多线程）"""
        key, question_data = item
        return self.analyze_question(key, question_data)
    
    def process_batch_multithread(self, items_to_process: List[tuple]) -> List[Dict[str, Any]]:
        """多线程处理批次"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(self.process_single_question, item): item 
                for item in items_to_process
            }
            
            # 收集结果
            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    item = future_to_item[future]
                    print(f"处理问题失败 {item[0]}: {e}")
        
        return results
    
    def run_analysis(self, use_multithread: bool = False) -> Dict[str, Any]:
        """运行完整分析"""
        self.load_data()
        
        # 限制处理的样本数量
        items_to_process = list(self.questions_data.items())
        if self.limit is not None and self.limit > 0:
            items_to_process = items_to_process[:self.limit]
            print(f"限制处理数量: 只处理前 {len(items_to_process)} 个样本")
        
        results = {
            'total_questions': len(self.questions_data),
            'processed_questions': len(items_to_process),
            'limit': self.limit,
            'margin': self.margin,
            'use_vlm': self.use_vlm,
            'max_workers': self.max_workers,
            'use_multithread': use_multithread,
            'analyses': []
        }
        
        print(f"开始分析，使用{'多线程' if use_multithread else '单线程'}模式...")
        start_time = time.time()
        
        if use_multithread and self.use_vlm:
            # 多线程处理
            print(f"使用多线程处理，最大线程数: {self.max_workers}")
            analyses = self.process_batch_multithread(items_to_process)
            results['analyses'] = analyses
        else:
            # 单线程处理（原有逻辑）
            if use_multithread and not self.use_vlm:
                print("警告: 未启用VLM，使用单线程模式")
            
            for question, question_data in items_to_process:
                analysis = self.analyze_question(question, question_data)
                if analysis:
                    results['analyses'].append(analysis)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"处理完成，耗时: {processing_time:.2f}秒")
        print(f"成功处理: {len(results['analyses'])}/{len(items_to_process)} 个问题")
        if processing_time > 0:
            speed = len(results['analyses']) / processing_time
            print(f"处理速度: {speed:.2f} 问题/秒")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存分析结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")
    
    def extract_ranking_pairs(self, ranking: str) -> set:
        """提取排序中的所有优劣关系对"""
        if not ranking or ranking in ["未获取", "VLM调用失败", "无法排序", "图片不存在"]:
            return set()
        
        try:
            # 按|分割成不同档次
            tiers = ranking.split('|')
            pairs = set()
            
            # 对于每两个不同档次，前面档次的所有元素都优于后面档次的所有元素
            for i, tier1 in enumerate(tiers):
                for j, tier2 in enumerate(tiers[i+1:], i+1):
                    for a in tier1:
                        for b in tier2:
                            if a in 'ABCD' and b in 'ABCD':
                                pairs.add((a, b))
            
            return pairs
        except Exception:
            return set()
    
    def check_semantic_consistency(self, score_ranking: str, vlm_ranking: str) -> tuple[bool, str]:
        """检查完全语义一致性（基于优劣关系对）"""
        # 提取两个排序的优劣关系对
        score_pairs = self.extract_ranking_pairs(score_ranking)
        vlm_pairs = self.extract_ranking_pairs(vlm_ranking)
        
        # 检查是否都包含有效的排序信息
        if not score_pairs or not vlm_pairs:
            return False, "无效排序"
        
        # 检查优劣关系对是否完全一致
        if score_pairs == vlm_pairs:
            return True, "语义一致"
        else:
            # 计算差异
            only_in_score = score_pairs - vlm_pairs
            only_in_vlm = vlm_pairs - score_pairs
            
            diff_info = []
            if only_in_score:
                diff_info.append(f"分数排序独有: {sorted(only_in_score)}")
            if only_in_vlm:
                diff_info.append(f"VLM排序独有: {sorted(only_in_vlm)}")
            
            return False, f"语义不一致 ({'; '.join(diff_info)})"

    def print_summary(self, results: Dict[str, Any]):
        """打印分析摘要"""
        print("\n" + "="*80)
        print("分析摘要")
        print("="*80)
        
        print(f"总问题数: {results['total_questions']}")
        if results['limit'] is not None:
            print(f"处理数量限制: {results['limit']}")
        print(f"实际处理数: {results.get('processed_questions', results['total_questions'])}")
        print(f"分档阈值(margin): {results.get('margin', 0.0)}")
        print(f"使用VLM排序: {'是' if results['use_vlm'] else '否'}")
        print(f"多线程模式: {'是' if results.get('use_multithread', False) else '否'}")
        if results.get('use_multithread', False):
            print(f"最大线程数: {results.get('max_workers', 'N/A')}")
        
        if results['analyses']:
            print(f"成功分析的问题数: {len(results['analyses'])}")
            
            # 统计排序一致性
            strict_consistent_count = 0  # 严格字符串一致性
            semantic_consistent_count = 0  # 语义一致性
            total_comparisons = 0
            
            for analysis in results['analyses']:
                score_ranking = analysis['score_ranking']
                vlm_ranking = analysis['vlm_ranking']
                
                print(f"\n问题: {analysis['question'][:50]}...")
                print(f"  分数排序: {score_ranking}")
                print(f"  VLM排序:  {vlm_ranking}")
                
                if vlm_ranking != "未获取" and vlm_ranking != "VLM调用失败":
                    # 验证排序结果是否有效（包含所有ABCD字母）
                    valid_chars = set(vlm_ranking.replace('|', ''))
                    if len(valid_chars) == 4 and valid_chars == {'A', 'B', 'C', 'D'}:
                        total_comparisons += 1
                        
                        # 严格一致性检查
                        strict_consistent = (score_ranking == vlm_ranking)
                        if strict_consistent:
                            strict_consistent_count += 1
                        
                        # 语义一致性检查
                        semantic_consistent, consistency_info = self.check_semantic_consistency(score_ranking, vlm_ranking)
                        if semantic_consistent:
                            semantic_consistent_count += 1
                        
                        # 显示一致性结果
                        if strict_consistent:
                            print(f"  严格一致性: ✓  语义一致性: ✓")
                        elif semantic_consistent:
                            print(f"  严格一致性: ✗  语义一致性: ✓")
                        else:
                            print(f"  严格一致性: ✗  语义一致性: ✗ ({consistency_info})")
            
            if total_comparisons > 0:
                strict_rate = strict_consistent_count / total_comparisons
                semantic_rate = semantic_consistent_count / total_comparisons
                print(f"\n一致性统计:")
                print(f"  严格一致性: {strict_consistent_count}/{total_comparisons} ({strict_rate:.1%})")
                print(f"  语义一致性: {semantic_consistent_count}/{total_comparisons} ({semantic_rate:.1%})")
                print(f"  语义提升: +{semantic_consistent_count - strict_consistent_count} 个样本")

def main():
    parser = argparse.ArgumentParser(description='简化的ABCD答案排序分析脚本')
    parser.add_argument('--input', '-i', required=True, help='输入JSONL文件路径')
    parser.add_argument('--output', '-o', default='abcd_ranking_results.json', help='输出JSON文件路径')
    parser.add_argument('--model-url', type=str, default=None, help='模型API地址')
    parser.add_argument('--model', type=str, default=None, help='模型名称')
    parser.add_argument('--api-key', type=str, default=None, help='API密钥')
    parser.add_argument('--use_img', action='store_true', help='是否使用图片进行VLM排序（默认True）')
    parser.add_argument('--limit', type=int, default=None, help='限制处理的样本数量（只处理前N个question_img）')
    parser.add_argument('--margin', type=float, default=0.0, help='分档阈值，分数差小于等于margin的归为同一档（默认0.0，不使用分档）')
    parser.add_argument('--max_workers', type=int, default=4, help='多线程处理的最大线程数（默认4）')
    parser.add_argument('--use_multithread', action='store_true', help='是否使用多线程处理（仅在启用VLM时有效）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 运行分析
    analyzer = SimpleABCDAnalyzer(
        args.input,
        model_url=args.model_url,
        model=args.model,
        api_key=args.api_key,
        use_img=args.use_img,
        limit=args.limit,
        margin=args.margin,
        max_workers=args.max_workers
    )
    results = analyzer.run_analysis(use_multithread=args.use_multithread)
    
    # 保存和显示结果
    analyzer.save_results(results, args.output)
    analyzer.print_summary(results)

if __name__ == "__main__":
    main()
# 示例命令:
'''
python /data/dlf/code/Field-Fidelity/tools/filter_rm_reasoing/simple_abcd_ranking.py \
  --input /data/dlf/code/Field-Fidelity/outputs/results/4o_rm_20251024_095439_sky_rm.jsonl \
  --output /data/dlf/code/Field-Fidelity/outputs/results/rm_abcd_results_margin_mt_text.json \
  --model-url https://api.zjuqx.cn/v1 \
  --model  anthropic.claude-3-5-sonnet-20241022-v2:0 \
  --api-key sk-DyIxTyw8hEOt8Wxo75D64989687044E28cE3Db73F735C09f \
  --limit 8 \
  --margin 0.3 \
  --use_multithread \
  --max_workers 8


# 多线程模式（推荐）:
python /data/dlf/code/Field-Fidelity/tools/filter_rm_reasoing/simple_abcd_ranking.py \
  --input /data/dlf/code/Field-Fidelity/outputs/results/4o_rm_20251024_095439_atomic_vl_mt.jsonl \
  --output /data/dlf/code/Field-Fidelity/outputs/results/rm_abcd_results_margin_mt_text.json \
  --model-url http://10.160.199.226:8001/v1 \
  --model /data/share/hub/models/Qwen/Qwen3-VL-235B-A22B-Instruct \
  --api-key empty \
  --limit 10 \
  --margin 0.3 \
  --use_multithread \
  --max_workers 8
'''

