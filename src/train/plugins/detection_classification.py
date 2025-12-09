#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detection和Classification任务的Reward函数插件

统一管理不同任务的reward计算逻辑，支持：
1. format_reward: 检查输出格式是否符合要求
2. task_acc_reward: 根据任务类型（detection/classification）计算准确度
   - detection: 计算IoU和Confidence
   - classification: 计算字符串匹配
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from swift.plugin import ORM, orms



# ============================================================================
# Detection任务相关函数
# ============================================================================

def extract_bbox(response: str) -> Optional[List[Dict]]:
    """
    从response中提取bounding box信息
    
    Args:
        response: 模型输出的字符串
    
    Returns:
        bbox列表，格式: [{'Position': [x1,y1,x2,y2], 'Confidence': conf}, ...]
    """
    start_tag = "<answer>"
    end_tag = "</answer>"
    input_str = response
    bbox_list = []
    # Check if the start tag is in the string
    if start_tag in input_str:
        # Extract the content between the start tag and end tag
        start_idx = input_str.find(start_tag) + len(start_tag)
        end_idx = input_str.find(end_tag)
        
        # If end_tag is not found (i.e., the string is truncated), assume it should be at the end
        if end_idx == -1:
            end_idx = len(input_str)
    
        content_str = input_str[start_idx:end_idx]
    
        # Check if it ends with a closing bracket, if not, fix it
        if not content_str.endswith("]"):
            # If the string is truncated, remove the incomplete part
            content_str = content_str.rsplit("},", 1)[0] + "}]"
    
        # Replace single quotes with double quotes for valid JSON
        content_str_corrected = content_str.replace("'", '"')
    
        # Convert the corrected string to a list of dictionaries (JSON format)
        try:
            bbox_list = json.loads(content_str_corrected)
        except json.JSONDecodeError as e:
            bbox_list = None
    else:
        bbox_list = None
    return bbox_list


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    计算两个bounding box的IoU
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU值 (0-1)
    """
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou


def sort_and_calculate_iou(list1: List[Dict], list2: List[Dict], 
                           iou_threshold: float = 0.5) -> List[tuple]:
    """
    对预测框按confidence排序，并与ground truth计算IoU
    
    Args:
        list1: ground truth bboxes
        list2: predicted bboxes
        iou_threshold: IoU阈值
    
    Returns:
        [(iou, confidence), ...] 列表
    """
    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    iou_results = []
    matched_list1_indices = set()

    for bbox2 in list2_sorted:
        max_iou = 0
        matched_bbox1 = -1
        best_iou = 0
        for i, bbox1 in enumerate(list1):
            if i not in matched_list1_indices:
                iou = calculate_iou(bbox1['Position'], bbox2['Position'])
                if iou > best_iou:
                    best_iou = iou
                    matched_bbox1 = i

        if best_iou > iou_threshold:
            iou_results.append((best_iou, bbox2['Confidence']))
            matched_list1_indices.add(matched_bbox1)
        else:
            iou_results.append((0, bbox2['Confidence']))
    
    return iou_results


def remove_duplicates(bbox_list: List[Dict]) -> List[Dict]:
    """去除重复的bounding box"""
    seen = set()
    unique_bboxes = []
    
    for bbox in bbox_list:
        # Convert the position tuple to a tuple for set hashing
        position_tuple = tuple(bbox['Position'])
        
        if position_tuple not in seen:
            seen.add(position_tuple)
            unique_bboxes.append(bbox)
    
    return unique_bboxes


def compute_reward_iou_v2(iou_results: List[tuple], len_gt: int) -> float:
    """
    基于IoU计算reward (V2版本)
    
    Args:
        iou_results: [(iou, confidence), ...] 列表
        len_gt: ground truth的数量
    
    Returns:
        reward值 (0-1)
    """
    iou_reward = 0.0
    confidence_reward = 0.0
    
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1 - temp_iou) * (1 - temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    if len_gt >= len(iou_results):
        iou_reward = iou_reward / len_gt
    else:
        iou_reward = iou_reward / len(iou_results)
    
    return iou_reward


def compute_reward_confidence(iou_results: List[tuple]) -> float:
    """
    基于Confidence计算reward
    
    Args:
        iou_results: [(iou, confidence), ...] 列表
    
    Returns:
        reward值 (0-1)
    """
    iou_reward = 0.0
    confidence_reward = 0.0
    
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1 - temp_iou) * (1 - temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    iou_reward = iou_reward / len(iou_results)
    confidence_reward = confidence_reward / len(iou_results)
    
    return confidence_reward


# ============================================================================
# Classification任务相关函数
# ============================================================================

def extract_classification_answer(text: str) -> str:
    """
    从文本中提取classification答案
    
    Args:
        text: 模型输出或ground truth文本
    
    Returns:
        提取的答案字符串
    """
    # Extract answer from text if it has think/answer tags
    match = re.search(r'<answer>(.*?)</answer>', text)
    answer = match.group(1).strip() if match else text.strip()
    
    # 标准化：去除空格、下划线，转小写
    answer = answer.replace(' ', '').replace('_', '').lower()
    
    return answer


def compute_classification_reward(model_answer: str, ground_truth: str) -> float:
    """
    计算classification的reward
    
    使用字符串包含关系判断：
    - 如果model_answer包含ground_truth，或ground_truth包含model_answer，则reward=1
    - 否则reward=0
    
    Args:
        model_answer: 模型答案
        ground_truth: 标准答案
    
    Returns:
        reward值 (0 or 1)
    """
    if ground_truth in model_answer or model_answer in ground_truth:
        return 1.0
    return 0.0


# ============================================================================
# 任务类型判断
# ============================================================================

def get_task_type(example: Dict[str, Any]) -> str:
    """
    根据样本特征判断任务类型
    
    判断逻辑：
    1. 如果有image_path字段，检查路径中是否包含'classification'
    2. 否则，检查solution格式：
       - 如果包含bbox格式（Position, Confidence），判断为detection
       - 否则判断为classification
    
    Args:
        example: 样本字典，包含image_path, problem, solution等字段
    
    Returns:
        'detection' 或 'classification'
    """
    # 方法1：通过image_path判断
    if 'image_path' in example:
        image_path = example['image_path']
        if 'classification' in image_path.lower():
            return 'classification'
        elif 'detection' in image_path.lower():
            return 'detection'
    
    # 方法2：通过solution格式判断
    if 'solution' in example:
        solution = example['solution']
        # 检查是否包含bbox格式
        if 'Position' in solution and 'Confidence' in solution:
            return 'detection'
        # 检查是否为bbox的JSON格式
        try:
            bbox_data = extract_bbox(solution)
            if bbox_data and isinstance(bbox_data, list) and len(bbox_data) > 0:
                if isinstance(bbox_data[0], dict) and 'Position' in bbox_data[0]:
                    return 'detection'
        except:
            pass
    
    # 默认为classification
    return 'classification'





class FormatReward(ORM):
    """
    格式检查reward：检查输出是否符合 <think>...</think><answer>...</answer> 格式
    """
    
    def __init__(self):
        """初始化格式检查器"""
        self.pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    
    def __call__(self, inputs: List[Dict], **kwargs) -> List[float]:
        """
        检查completion格式
        
        Args:
            inputs: 输入数据列表，每个元素包含messages等字段
            **kwargs: 其他参数
        
        Returns:
            reward列表，每个元素为0.0或1.0
        """
        rewards = []
        
        for inp in inputs:
            # 从messages中提取assistant的回答
            messages = inp.get('messages', [])
            content = ""
            
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    content = str(msg.get('content', ''))
                    break
            
            # 检查格式
            match = re.fullmatch(self.pattern, content, re.DOTALL)
            rewards.append(1.0 if match else 0.0)
        
        return rewards


class TaskAccReward(ORM):
    """
    任务准确度reward：根据任务类型（detection/classification）计算不同的reward
    
    对于detection任务：
    - 计算IoU reward和Confidence reward的加权平均
    
    对于classification任务：
    - 计算字符串匹配的准确度
    """
    
    def __init__(self, iou_weight: float = 0.6, conf_weight: float = 0.4):
        """
        初始化任务准确度评估器
        
        Args:
            iou_weight: IoU reward的权重（detection任务）
            conf_weight: Confidence reward的权重（detection任务）
        """
        self.iou_weight = iou_weight
        self.conf_weight = conf_weight
    
    def __call__(self, inputs: List[Dict], **kwargs) -> List[float]:
        """
        计算任务准确度reward
        
        Args:
            inputs: 输入数据列表，每个元素包含messages, reference/solution等字段
            **kwargs: 其他参数
        
        Returns:
            reward列表
        """
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        
        for idx, inp in enumerate(inputs):
            # 提取信息
            messages = inp.get('messages', [])
            content = ""
            
            # 提取assistant的回答
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    content = str(msg.get('content', ''))
                    break
            
            # 获取参考答案（ground truth）
            sol = inp.get('reference') or inp.get('solution') or ''
            
            # 判断任务类型
            task_type = get_task_type(inp)
            
            reward = 0.0
            debug_info = {}
            
            if task_type == 'detection':
                # ============ Detection任务 ============
                try:
                    ground_truth = sol.strip()
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    model_answer = content_match.group(1).strip() if content_match else content.strip()
                    #model_answer = '<answer>' + model_answer + '</answer>'

                    # Fix format errors
                    model_answer = model_answer.replace("[[", '[')
                    model_answer = model_answer.replace("]]", ']')
                    model_answer = model_answer.replace("\n", '')
                    
                    ground_truth_bbox = extract_bbox(ground_truth)
                    model_answer_bbox = extract_bbox(model_answer)
                    
                    if (ground_truth_bbox is None or model_answer_bbox is None or 
                        len(model_answer_bbox) == 0 or type(model_answer_bbox[0]) != dict):
                        reward = 0.0
                    else:
                        model_answer_bbox = remove_duplicates(model_answer_bbox)
                        iou_results = sort_and_calculate_iou(ground_truth_bbox, model_answer_bbox)
                        
                        # 计算IoU reward和Confidence reward的加权平均
                        iou_reward = compute_reward_iou_v2(iou_results, len(ground_truth_bbox))
                        confidence_reward = compute_reward_confidence(iou_results)
                        
                        # 使用配置的权重
                        reward = self.iou_weight * iou_reward + self.conf_weight * confidence_reward
                        
                        if reward > 1:
                            reward = 1.0
                        
                        debug_info = {
                            'task': 'detection',
                            'model_bbox': model_answer_bbox,
                            'ground_truth_bbox': ground_truth_bbox,
                            'iou_results': iou_results,
                            'iou_reward': iou_reward,
                            'confidence_reward': confidence_reward
                        }
                        
                except Exception as e:
                    reward = 0.0
                    debug_info = {'task': 'detection', 'error': str(e)}
            
            else:
                # ============ Classification任务 ============
                try:
                    ground_truth = extract_classification_answer(sol)
                    model_answer = extract_classification_answer(content)
                    
                    reward = compute_classification_reward(model_answer, ground_truth)
                    
                    debug_info = {
                        'task': 'classification',
                        'model_answer': model_answer,
                        'ground_truth': ground_truth
                    }
                    
                except Exception as e:
                    reward = 0.0
                    debug_info = {'task': 'classification', 'error': str(e)}
            
            rewards.append(reward)
            
            # Debug日志
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                if log_path:
                    with open(log_path, "a") as f:
                        f.write(f"------------- {current_time} Task Accuracy Reward: {reward} -------------\n")
                        f.write(f"Task Type: {task_type}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n")
                        f.write(f"Debug Info: {debug_info}\n")
                        f.write("\n")
        
        return rewards



orms['det_cls_format'] = FormatReward
orms['det_cls_acc'] = TaskAccReward







