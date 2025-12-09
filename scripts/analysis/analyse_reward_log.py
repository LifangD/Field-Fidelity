import json
from collections import defaultdict
from typing import Dict, List
import pandas as pd
import os

# 文件路径
path = "/data/dlf/code/Field-Fidelity/outputs/experiments/gspo_fvit/reward_anything/reward_evaluation_20251127_114216/reward_details.jsonl"

# 训练数据路径
training_data_paths = [
    "/data/dlf/code/Field-Fidelity/data/rlhf/formatted/rlhf_formatted.jsonl",
    "/data/dlf/code/Field-Fidelity/data/idk/data_format/idk_train_formatted_1k.jsonl"
]

def load_question_to_image_mapping(training_data_paths: List[str]) -> Dict[str, str]:
    """从训练数据中加载question到image的映射"""
    
    print("\n正在加载训练数据以获取图片映射...")
    question_to_image = {}
    
    for data_path in training_data_paths:
        if not os.path.exists(data_path):
            print(f"警告: 文件不存在 {data_path}")
            continue
        
        print(f"  加载: {data_path}")
        count = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # 从messages中提取question
                    if 'messages' in data and len(data['messages']) > 0:
                        # 提取用户的最后一个问题
                        user_messages = [msg for msg in data['messages'] if msg['role'] == 'user']
                        if user_messages:
                            question_with_image = user_messages[-1]['content']
                            # 去除<image>标记
                            question_clean = question_with_image.replace('<image>', '').strip()
                            
                            if 'images' in data:
                                question_to_image[question_clean] = data['images']
                                count += 1
                except Exception as e:
                    continue
        
        print(f"    成功加载 {count} 条问题-图片映射")
    
    print(f"  总共加载 {len(question_to_image)} 条唯一的问题-图片映射\n")
    return question_to_image


def load_and_analyze_data(file_path: str, total_lines: int = 122211, num_epochs: int = 3):
    """加载数据并按epoch组织"""
    
    # 估算每个epoch的行数
    lines_per_epoch = total_lines // num_epochs
    
    print(f"总行数: {total_lines}")
    print(f"预估每个epoch行数: {lines_per_epoch}")
    print(f"Epoch划分:")
    print(f"  Epoch 1: 行 1-{lines_per_epoch}")
    print(f"  Epoch 2: 行 {lines_per_epoch+1}-{lines_per_epoch*2}")
    print(f"  Epoch 3: 行 {lines_per_epoch*2+1}-{total_lines}")
    print("\n正在加载数据...")
    
    # 存储每个epoch中每个question的数据
    epoch_data = {1: {}, 2: {}, 3: {}}
    
    error_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"已处理 {line_num}/{total_lines} 行...")
            
            try:
                data = json.loads(line.strip())
                question = data['question']
                all_scores = data['all_scores']
                response_key = data.get('response_key', '')
                current_answer = data.get('current_answer', '')
                reference_answer = data.get('reference_answer', '')
                reasoning = data.get('reasoning', '')
                
                # 确定当前属于哪个epoch
                if line_num <= lines_per_epoch:
                    epoch = 1
                elif line_num <= lines_per_epoch * 2:
                    epoch = 2
                else:
                    epoch = 3
                
                # 只保存每个question的第一次出现
                if question not in epoch_data[epoch]:
                    epoch_data[epoch][question] = {
                        'all_scores': all_scores,
                        'responses': {}
                    }
                
                # 保存每个response的详细信息
                if response_key and response_key not in epoch_data[epoch][question]['responses']:
                    epoch_data[epoch][question]['responses'][response_key] = {
                        'current_answer': current_answer,
                        'reference_answer': reference_answer,
                        'reasoning': reasoning
                    }
            except json.JSONDecodeError as e:
                error_count += 1
                if error_count <= 10:
                    print(f"警告: 第 {line_num} 行JSON解析错误，跳过: {str(e)[:100]}")
                continue
            except KeyError as e:
                error_count += 1
                if error_count <= 10:
                    print(f"警告: 第 {line_num} 行缺少字段 {e}，跳过")
                continue
    
    if error_count > 0:
        print(f"\n共跳过 {error_count} 个错误行")
    
    return epoch_data


def find_common_questions(epoch_data: Dict) -> List[str]:
    """找出在所有epoch中都出现的question"""
    
    questions_epoch1 = set(epoch_data[1].keys())
    questions_epoch2 = set(epoch_data[2].keys())
    questions_epoch3 = set(epoch_data[3].keys())
    
    common_questions = questions_epoch1 & questions_epoch2 & questions_epoch3
    
    print(f"\nEpoch 1 问题数: {len(questions_epoch1)}")
    print(f"Epoch 2 问题数: {len(questions_epoch2)}")
    print(f"Epoch 3 问题数: {len(questions_epoch3)}")
    print(f"所有epoch共有的问题数: {len(common_questions)}")
    
    return list(common_questions)


def extract_sample_questions(epoch_data: Dict, common_questions: List[str], 
                            question_to_image: Dict[str, str], num_samples: int = 30):
    """提取样本问题的数据"""
    
    # 选择前N个问题
    sample_questions = common_questions[:num_samples]
    
    results = []
    
    for question in sample_questions:
        result = {
            'question': question,
            'image_path': question_to_image.get(question, 'N/A'),
            'epochs': {}
        }
        
        for epoch in [1, 2, 3]:
            question_data = epoch_data[epoch][question]
            result['epochs'][f'epoch_{epoch}'] = {
                'scores': question_data['all_scores'],
                'responses_details': question_data['responses']
            }
        
        results.append(result)
    
    return results


def format_results(results: List[Dict]):
    """格式化并输出结果"""
    
    print("\n" + "="*100)
    print(f"选取的{len(results)}个问题及其在3个epoch中的评分情况")
    print("="*100)
    
    for i, result in enumerate(results, 1):
        print(f"\n{'='*100}")
        print(f"问题 {i}:")
        print(f"{'='*100}")
        print(f"Question: {result['question']}")
        print(f"Image: {result['image_path']}")
        print(f"\n{'─'*100}")
        
        # 创建一个表格展示
        responses = ['response_a', 'response_b', 'response_c', 'response_d', 
                    'response_e', 'response_f', 'response_g', 'response_h']
        
        print(f"\n{'Response':<15} {'Epoch 1':<10} {'Epoch 2':<10} {'Epoch 3':<10} {'变化':<15}")
        print(f"{'-'*60}")
        
        for response in responses:
            score_epoch1 = result['epochs']['epoch_1']['scores'].get(response, 'N/A')
            score_epoch2 = result['epochs']['epoch_2']['scores'].get(response, 'N/A')
            score_epoch3 = result['epochs']['epoch_3']['scores'].get(response, 'N/A')
            
            # 计算变化
            if all(isinstance(s, (int, float)) for s in [score_epoch1, score_epoch2, score_epoch3]):
                change = f"{score_epoch3 - score_epoch1:+.1f}"
            else:
                change = "N/A"
            
            print(f"{response:<15} {str(score_epoch1):<10} {str(score_epoch2):<10} {str(score_epoch3):<10} {change:<15}")


def save_to_csv(results: List[Dict], output_path: str = "reward_analysis.csv"):
    """保存结果到CSV文件，每行代表一个question+epoch+response组合"""
    
    rows = []
    for i, result in enumerate(results, 1):
        responses = ['response_a', 'response_b', 'response_c', 'response_d', 
                    'response_e', 'response_f', 'response_g', 'response_h']
        
        # 为每个epoch的每个response创建一行
        for epoch in [1, 2, 3]:
            epoch_key = f'epoch_{epoch}'
            epoch_data = result['epochs'][epoch_key]
            
            for response in responses:
                row = {
                    'question_id': i,
                    'question': result['question'],
                    'image_path': result['image_path'],
                    'epoch': epoch,
                    'response_key': response
                }
                
                # 添加该response在该epoch的评分
                score = epoch_data['scores'].get(response, None)
                row['score'] = score
                
                # 添加该response在该epoch的详细信息
                if response in epoch_data['responses_details']:
                    details = epoch_data['responses_details'][response]
                    row['current_answer'] = details.get('current_answer', '')
                    row['reference_answer'] = details.get('reference_answer', '')
                    row['reasoning'] = details.get('reasoning', '')
                else:
                    row['current_answer'] = ''
                    row['reference_answer'] = ''
                    row['reasoning'] = ''
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    # 调整列顺序，让主键列在前面
    column_order = ['question_id', 'question', 'image_path', 'epoch', 'response_key', 'score', 
                    'current_answer', 'reference_answer', 'reasoning']
    df = df[column_order]
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n结果已保存到: {output_path}")


def generate_analysis_report(results: List[Dict], output_path: str = "reward_analysis_report.txt"):
    """生成详细的分析报告"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("Reward Model 训练分析报告\n")
        f.write("="*100 + "\n\n")
        
        f.write("## 数据概览\n\n")
        f.write(f"- 分析的问题数量: {len(results)}\n")
        f.write(f"- Epoch数量: 3\n")
        f.write(f"- 每个问题的response数量: 8 (response_a ~ response_h)\n\n")
        
        responses = ['response_a', 'response_b', 'response_c', 'response_d', 
                    'response_e', 'response_f', 'response_g', 'response_h']
        
        # 统计总体变化趋势
        f.write("## 总体评分变化趋势\n\n")
        total_improvements = 0
        total_declines = 0
        total_stable = 0
        
        for result in results:
            for response in responses:
                try:
                    score_epoch1 = result['epochs']['epoch_1']['scores'].get(response)
                    score_epoch3 = result['epochs']['epoch_3']['scores'].get(response)
                    
                    if score_epoch1 is not None and score_epoch3 is not None:
                        if score_epoch3 > score_epoch1:
                            total_improvements += 1
                        elif score_epoch3 < score_epoch1:
                            total_declines += 1
                        else:
                            total_stable += 1
                except:
                    continue
        
        total_valid = total_improvements + total_declines + total_stable
        f.write(f"- 评分提升的response: {total_improvements} ({total_improvements/total_valid*100:.1f}%)\n")
        f.write(f"- 评分下降的response: {total_declines} ({total_declines/total_valid*100:.1f}%)\n")
        f.write(f"- 评分保持不变的response: {total_stable} ({total_stable/total_valid*100:.1f}%)\n\n")
        
        # 详细问题分析
        f.write("="*100 + "\n")
        f.write("## 详细问题分析\n")
        f.write("="*100 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"\n{'='*100}\n")
            f.write(f"问题 {i}\n")
            f.write(f"{'='*100}\n")
            f.write(f"Question: {result['question']}\n")
            f.write(f"Image: {result['image_path']}\n\n")
            
            # 评分表格
            f.write(f"{'Response':<15} {'Epoch 1':<10} {'Epoch 2':<10} {'Epoch 3':<10} {'变化':<10} {'趋势':<10}\n")
            f.write(f"{'-'*75}\n")
            
            for response in responses:
                score_epoch1 = result['epochs']['epoch_1']['scores'].get(response, 'N/A')
                score_epoch2 = result['epochs']['epoch_2']['scores'].get(response, 'N/A')
                score_epoch3 = result['epochs']['epoch_3']['scores'].get(response, 'N/A')
                
                if all(isinstance(s, (int, float)) for s in [score_epoch1, score_epoch2, score_epoch3]):
                    change = score_epoch3 - score_epoch1
                    change_str = f"{change:+.0f}"
                    
                    if change > 0:
                        trend = "↑ 提升"
                    elif change < 0:
                        trend = "↓ 下降"
                    else:
                        trend = "→ 稳定"
                else:
                    change_str = "N/A"
                    trend = "N/A"
                
                f.write(f"{response:<15} {str(score_epoch1):<10} {str(score_epoch2):<10} {str(score_epoch3):<10} {change_str:<10} {trend:<10}\n")
            
            f.write("\n")
    
    print(f"详细分析报告已保存到: {output_path}")


def main():
    # 加载训练数据中的问题-图片映射
    question_to_image = load_question_to_image_mapping(training_data_paths)
    
    # 加载数据
    epoch_data = load_and_analyze_data(path)
    
    # 找出共同的问题
    common_questions = find_common_questions(epoch_data)
    
    if len(common_questions) < 30:
        print(f"\n警告: 只找到 {len(common_questions)} 个共同问题，少于请求的30个")
        num_samples = len(common_questions)
    else:
        num_samples = 30
    
    # 提取样本
    results = extract_sample_questions(epoch_data, common_questions, question_to_image, num_samples)
    
    # 格式化输出
    format_results(results)
    
    # 保存到CSV
    save_to_csv(results)
    
    # 生成详细报告
    generate_analysis_report(results)
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()