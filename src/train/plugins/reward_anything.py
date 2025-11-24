import re
import json
import os
import logging
from datetime import datetime
from typing import List, Dict
import torch
from swift.plugin import ORM, orms, rm_plugins
from swift.utils import get_logger

try:
    import rewardanything
    REWARDANYTHING_AVAILABLE = True
except ImportError:
    print("警告: rewardanything库未安装")
    REWARDANYTHING_AVAILABLE = False
    
os.environ['no_proxy'] = 'localhost,127.0.0.1'
# 设置日志
logger = get_logger()

def setup_reward_logging():
    """设置reward评估日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.environ["OUTPUT_DIR"]
    log_dir = f"{out_dir}/reward_evaluation_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建reward评估专用logger
    reward_logger = logging.getLogger('reward_evaluation')
    reward_logger.setLevel(logging.INFO)
    
    # 文件handler
    handler = logging.FileHandler(
        os.path.join(log_dir, 'reward_evaluation.log'),
        encoding='utf-8'
    )
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    reward_logger.addHandler(handler)
    
    # 创建JSONL日志文件路径
    jsonl_path = os.path.join(log_dir, 'reward_details.jsonl')
    
    reward_logger.info("=" * 60)
    reward_logger.info(f"日志目录: {log_dir}")
    reward_logger.info(f"JSONL文件: {jsonl_path}")
    reward_logger.info("=" * 60)
    
    return reward_logger, jsonl_path

def write_reward_record(jsonl_path: str, record: dict):
    """写入奖励评估记录到JSONL文件"""
    try:
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"写入JSONL记录失败: {e}")

# 初始化日志系统
reward_logger, REWARD_JSONL_PATH = setup_reward_logging()


def contains_idk(text):
    KEYWORDS = [
        "ambiguous",
        "bad question", 
        "cannot confirm",
        "depend",
        "don't know",
        "it is difficult",
        "i can't",
        "none",
        "not clear",
        "not sure", 
        "sorry",
        "hard to determine",
        "not possible",
        "uncertain",
        "unanswerable",
        "unknown",
        "not certain",
        "cannot determine"
        
    ]

    if not text:
        return False
        
    text_lower = text.lower().strip()
    
    for keyword in KEYWORDS:
        if keyword.lower() in text_lower:
            return True
            
    return False

class IDKGenRM(ORM):

    
    def __init__(self, model=None, template=None, api_base_url="http://localhost:8001"):
        self.model = model
        self.template = template
        self.api_base_url = api_base_url
        self.debug_mode = False
        
        # 初始化 RewardAnything Client
        if REWARDANYTHING_AVAILABLE:
            try:
                self.ra_client = rewardanything.Client(api_base_url)
                logger.info(f"成功初始化 RewardAnything Client: {api_base_url}")
            except Exception as e:
                logger.error(f"初始化 RewardAnything Client 失败: {e}")
                self.ra_client = None
        else:
            logger.warning("rewardanything 库未安装，功能受限")
            self.ra_client = None
        
        # 系统提示（保留以便兼容）
        self.system_prompt = self._build_system_prompt()
        self.reward_prompt = self._build_skyrm_question_template()



    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return "You are a helpful assistant"
    def _build_skyrm_question_template(self) -> str:
        """导入奖励提示"""
        with open("/data/dlf/code/Field-Fidelity/src/train/prompt/sky_rm_v2.md", "r") as f:
            return f.read()

        


    def extract_question_and_answer(self, inp: Dict) -> tuple:
        """从输入中提取问题、回答和参考答案"""
        messages = inp.get('messages', [])
        current_answer = ""
        question = ""
        
        # 提取问题
        if messages and len(messages) >= 2:
            for msg in reversed(messages):
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        question = content
                    elif isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                        question = ' '.join(text_parts)
                    break
            
            # 提取回答
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    current_answer = str(msg.get('content', ''))
                    break
        
        # 清理问题
        if "<image>" in question:
            question = question.replace("<image>", "")
        
        # 提取参考答案
        reference_answer = inp.get('reference') or inp.get('solution') or ''
        
        # 提取答案内容（去掉think标签，只保留answer部分）
        rm_current_answer = current_answer
        if "<answer>" in current_answer and "</answer>" in current_answer:
            matches = re.findall("<answer>(.*?)</answer>", current_answer, re.DOTALL)
            if len(matches) > 0:
                rm_current_answer = matches[0]
        
        return question, rm_current_answer, reference_answer, current_answer

    def group_inputs_by_question(self, inputs: List[Dict]) -> Dict[str, List[tuple]]:
        """将inputs按问题分组，返回 {question_key: [(idx, answer, reference, raw_answer), ...]}"""
        question_groups = {}
        
        for idx, inp in enumerate(inputs):
            question, answer, reference, raw_answer = self.extract_question_and_answer(inp)
            
            if not question or not answer:
                # 如果缺少必要信息，创建一个唯一key避免分组
                question_key = f"__invalid_{idx}__"
            else:
                # 使用 question + reference 作为分组key
                question_key = f"{question}|||{reference}"
            
            if question_key not in question_groups:
                question_groups[question_key] = []
            
            question_groups[question_key].append((idx, answer, reference, raw_answer, question))
        
        return question_groups

    def __call__(self, inputs: List[Dict], **kwargs) -> torch.Tensor:
        """计算奖励分数 - 批量处理同一问题的多个响应"""
        trainer_state = kwargs.get('trainer_state')
        if trainer_state:
            global_step = getattr(trainer_state, 'global_step', 0)
            max_steps = getattr(trainer_state, 'max_steps', 0)
            logger.info(f"global_step: {global_step}, max_steps: {max_steps}")
        
        assert inputs is not None, "inputs is None"
        logger.info(f"开始处理批次: {len(inputs)}样本")
        
        # 检查 RewardAnything Client 是否可用
        if not (REWARDANYTHING_AVAILABLE and self.ra_client):
            logger.error("RewardAnything Client 未初始化，无法进行评分")
            return torch.tensor([0.0] * len(inputs), dtype=torch.float32)
            
        logger.info(f"使用评分方法: RewardAnything Client ({self.api_base_url})")
        reward_logger.info("=" * 60)
        reward_logger.info(f"开始新批次评估: {len(inputs)} 个样本")
        reward_logger.info(f"使用评分方法: RewardAnything Client ({self.api_base_url})")
        reward_logger.info("=" * 60)
        
        # 准备奖励分数列表
        rewards = [0.0] * len(inputs)
        
        # 按问题分组
        question_groups = self.group_inputs_by_question(inputs)
        logger.info(f"分组后有 {len(question_groups)} 个不同的问题")
        
        # 对每组问题批量评分
        for question_key, group_data in question_groups.items():
            try:
                # 检查是否为无效样本
                if question_key.startswith("__invalid_"):
                    idx = group_data[0][0]
                    logger.warning(f"样本 {idx + 1}: 缺少必要信息，跳过")
                    continue
                
                # 提取问题和参考答案
                question = group_data[0][4]  # question
                reference_answer = group_data[0][2]  # reference
                
                # 构造 responses 字典
                responses = {}
                idx_to_response_key = {}  # 映射 idx -> response_key
                
                for i, (idx, answer, ref, raw_answer, _) in enumerate(group_data):
                    response_key = f"response_{chr(97 + i)}"  # response_a, response_b, ...
                    responses[response_key] = answer
                    idx_to_response_key[idx] = response_key
                
                logger.info(f"问题有 {len(responses)} 个响应: {list(responses.keys())}")
                
                # 构造 principle
                principle='''
Evaluate responses using these criteria:

1. **Answer Accuracy** (40%): Factual correctness and alignment with reference answer when provided
2. **Appropriateness** (30%): Proper handling of certainty/uncertainty, appropriate use of IDK when knowledge is limited
3. **Expression Quality** (20%): Clear explanations, logical structure, and instruction following
4. **Relevance** (10%): Response relevance to the question and task requirements

For conflicting criteria, prioritize: accuracy > appropriateness > expression quality > relevance.

Special scoring rules:
- When reference answer exists: Correct answer > IDK/Uncertain > Wrong answer
- When reference answer is IDK: Must express uncertainty, specific answers receive penalty
- When no reference answer: Focus on instruction following and response reasonableness
'''
                if len(reference_answer) == 0:
                    prompt = question
                else:
                    prompt = f"Question: {question}\n\nReference Answer: {reference_answer}"
                
                # 批量评分
                try:
                    # 构造请求
                    request = {
                        "principle": principle,
                        "prompt": prompt,
                        "responses": responses
                    }
                    
                    # 调用 Client
                    results = self.ra_client.judge_batch([request])
                    
                    if results and len(results) > 0:
                        result = results[0]
                        # 获取分数字典
                        if hasattr(result, 'scores') and result.scores:
                            logger.info(f"获得评分: {result.scores}")
                            
                            # 将分数映射回原始idx
                            for idx, answer, ref, raw_answer, _ in group_data:
                                response_key = idx_to_response_key[idx]
                                if response_key in result.scores:
                                    score = float(result.scores[response_key])
                                    
                                    # # restrict idk semantic-score
                                    # if not contains_idk(reference_answer) and contains_idk(raw_answer) and score > 0:
                                    #     score = -score
                                    
                                    rewards[idx] = score
                                    logger.info(f"样本 {idx + 1} ({response_key}) 奖励: {score:.4f}")
                                    
                                    # 记录详细日志
                                    record = {
                                        "timestamp": datetime.now().isoformat(),
                                        "question": question,
                                        "current_answer": answer,
                                        "reference_answer": reference_answer,
                                        "reward_score": score,
                                        "response_key": response_key,
                                        "all_scores": result.scores
                                    }
                                    write_reward_record(REWARD_JSONL_PATH, record)
                                else:
                                    logger.warning(f"样本 {idx + 1}: 未找到对应的评分key {response_key}")
                        else:
                            logger.warning(f"未获得有效评分结果")
                    else:
                        logger.warning(f"Client返回空结果")
                        
                except Exception as e:
                    logger.error(f"批量评分失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
            except Exception as e:
                logger.error(f"处理问题组失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        logger.info(f"批次完成: 平均奖励 {avg_reward:.4f}")
        
        # 批次总结日志
        reward_logger.info("=" * 60)
        reward_logger.info(f"批次评估完成")
        reward_logger.info(f"总样本数: {len(inputs)}")
        reward_logger.info(f"平均奖励: {avg_reward:.4f}")
        reward_logger.info(f"奖励分布: 最小={min(rewards):.4f}, 最大={max(rewards):.4f}")
        reward_logger.info("=" * 60)
        
        return torch.tensor(rewards, dtype=torch.float32)
        





class CustomFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]

rm_plugins['idk_genrm'] = IDKGenRM 
rm_plugins['custom_format'] = CustomFormat
