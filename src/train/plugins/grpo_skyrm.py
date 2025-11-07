import re
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
import torch
import requests
from transformers import AutoTokenizer
from swift.plugin import ORM, orms, rm_plugins
from swift.utils import get_logger
try:
    from swift.llm import PtEngine, RequestConfig, InferRequest
    from swift.llm import get_model_tokenizer, get_template
    from swift import Swift
    SWIFT_AVAILABLE = True
except ImportError:
    print("警告: swift库未安装，pt_engine功能将不可用")
    SWIFT_AVAILABLE = False
from openai import OpenAI
os.environ['no_proxy'] = 'localhost,127.0.0.1'
# 设置日志
logger = get_logger()

def setup_reward_logging():
    """设置reward评估日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/data/dlf/code/Field-Fidelity/outputs/experiments/logs/reward_evaluation_{timestamp}"
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

class IDKGenRM(ORM):

    
    def __init__(self, model=None, template=None, use_api=True, api_base_url="http://10.160.199.227:8008/classify", api_key="EMPTY", reward_model_name="/data/share/hub/models/Skywork/Skywork-Reward-V2-Llama-3___1-8B-40M"):
        self.model = model
        self.template = template
        self.use_api = use_api  # 新增：选择使用API还是PT引擎
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.reward_model_name = reward_model_name
        self.debug_mode = False
        
        # 初始化tokenizer用于reward model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
            logger.info(f"成功初始化reward model tokenizer: {reward_model_name}")
        except Exception as e:
            logger.error(f"初始化reward model tokenizer失败: {e}")
            self.tokenizer = None
        
        # 系统提示
        self.system_prompt = self._build_system_prompt()
        self.reward_prompt = self._build_skyrm_question_template()

    def process_convs(self, conv, base_url, tokenizer, model_name_or_path):
        """处理对话并获取reward分数"""
        try:
            payload = {"model": model_name_or_path}
            convs_formatted = []
          
            conv_formatted = tokenizer.apply_chat_template(conv, tokenize=False)
            if tokenizer.bos_token is not None and conv_formatted.startswith(tokenizer.bos_token):
                conv_formatted = conv_formatted[len(tokenizer.bos_token):]
            convs_formatted.append(conv_formatted)

            payload.update({"text": convs_formatted})
           
            response = requests.post(base_url, json=payload)
            
            if response.status_code == 200:
                responses = response.json()
                return responses[0]["embedding"][0]
            else:
                logger.error(f"Reward API调用失败，状态码: {response.status_code}, 响应: {response.text}")
                return 0.5  # 默认分数
                
        except Exception as e:
            logger.error(f"获取reward分数失败: {e}")
            return 0.5  # 默认分数

    def infer_with_engine(self, messages: List[Dict]) -> str:
        """使用PT引擎进行推理"""
        if not self.engine:
            raise RuntimeError("PT引擎未初始化")
        
        try:
            # 使用InferRequest格式
            infer_request = InferRequest(messages=messages)
            results = self.engine.infer([infer_request], self.request_config, use_tqdm=False)
            
            if results and len(results) > 0:
                result = results[0]
                choices = getattr(result, 'choices', None)
                if choices and len(choices) > 0:
                    choice = choices[0]
                    message = getattr(choice, 'message', None)
                    if message:
                        content = getattr(message, 'content', '')
                        return content
            
            raise RuntimeError("PT引擎推理失败：未获得有效响应")
            
        except Exception as e:
            logger.error(f"PT引擎推理失败: {e}")
            raise

    def infer_with_api(self, messages: List[Dict]) -> str:
        """使用API进行推理"""
        if not self.api_client or not self.api_model_name:
            raise RuntimeError("API客户端未初始化")
        
        try:
            response = self.api_client.chat.completions.create(
                model=self.api_model_name,
                messages=messages,
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096
            )
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("API推理失败：响应内容为None")
                return content
            else:
                raise RuntimeError("API推理失败：响应中没有choices")
                
        except Exception as e:
            logger.error(f"API推理失败: {e}")
            raise

    def infer(self, messages: List[Dict]) -> str:
        """统一推理接口"""
        if self.use_api:
            return self.infer_with_api(messages)
        else:
            return self.infer_with_engine(messages)

    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return "You are a helpful assistant"
    def _build_skyrm_question_template(self) -> str:
        """导入奖励提示"""
        with open("/data/dlf/code/Field-Fidelity/src/train/prompt/sky_rm.md", "r") as f:
            return f.read()

        


    def __call__(self, inputs: List[Dict], **kwargs) -> torch.Tensor:
        """计算奖励分数 - 使用Skywork Reward Model API"""
        trainer_state = kwargs.get('trainer_state')
        if trainer_state:
            global_step = getattr(trainer_state, 'global_step', 0)
            max_steps = getattr(trainer_state, 'max_steps', 0)
            logger.info(f"global_step: {global_step}, max_steps: {max_steps}")
        
        # 检查tokenizer是否可用
        if not self.tokenizer:
            logger.error("Reward model tokenizer未初始化")
            return torch.tensor([0.5] * len(inputs), dtype=torch.float32)
        
        assert inputs is not None, "inputs is None"
        logger.info(f"开始处理批次: {len(inputs)}样本")
        logger.info(f"使用Skywork Reward Model API: {self.api_base_url}")
        reward_logger.info("=" * 60)
        reward_logger.info(f"开始新批次评估: {len(inputs)} 个样本")
        reward_logger.info(f"使用Skywork Reward Model API: {self.api_base_url}")
        reward_logger.info("=" * 60)
        
        # 准备奖励分数列表
        rewards = [0.5] * len(inputs)  # 默认分数
        
        for idx, inp in enumerate(inputs):
            try:
                # 提取信息
                messages = inp.get('messages', [])
                current_answer = ""
                question = ""
                
                # 提取问题和回答
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
                    
                    for msg in reversed(messages):
                        if msg.get('role') == 'assistant':
                            current_answer = str(msg.get('content', ''))
                            break
                
                # 清理问题
                if "<image>" in question:
                    question = question.replace("<image>", "")
                reference_answer = inp.get('reference') or inp.get('solution') or ''
                if current_answer and question:
                    # 构造对话格式
                   
                    rm_question = self.reward_prompt.format(question=question, reference_answer=reference_answer)
                    rm_current_answer = current_answer
                    if "<answer>" in current_answer and "</answer>" in current_answer:
                        matches = re.findall("<answer>(.*?)</answer>", current_answer, re.DOTALL)
                        if len(matches) > 0:
                            rm_current_answer = matches[0]
                    if rm_current_answer and len(rm_current_answer)>0: 
                        conv = [
                            {"role": "user", "content": rm_question}, 
                            {"role": "assistant", "content": rm_current_answer}
                        ]
                        
                        # 调用reward model API
                        score = self.process_convs(
                            conv, 
                            self.api_base_url, 
                            self.tokenizer, 
                            self.reward_model_name
                        )
                    else:
                        score = 0.5
                    rewards[idx] = score
                    logger.info(f"样本 {idx + 1} 奖励: {score:.4f}")
                    
                    # 记录详细日志
                    record = {
                        "timestamp": datetime.now().isoformat(),
                        "question": question,
                        "current_answer": rm_current_answer,
                        "reference_answer": reference_answer,
                        "reward_score": score,
                        "original_messages": messages
                    }
                    write_reward_record(REWARD_JSONL_PATH, record)
                    
                    reward_logger.info(f"样本 {idx + 1}: 问题长度: {len(question)}, 回答长度: {len(current_answer)}, 奖励: {score:.4f}")
                else:
                    logger.warning(f"样本 {idx + 1}: 缺少必要信息")
                    reward_logger.warning(f"样本 {idx + 1}: 缺少必要信息 - current_answer: {bool(current_answer)}, question: {bool(question)}")
                    
            except Exception as e:
                logger.error(f"处理样本 {idx + 1} 失败: {e}")
                reward_logger.error(f"处理样本 {idx + 1} 失败: {e}")
                # 保持默认分数0.5
        
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
