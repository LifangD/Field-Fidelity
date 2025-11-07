import re
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
import torch
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

    
    def __init__(self, model=None, template=None, use_api=True, api_base_url="http://10.160.199.226:8001/v1", api_key="EMPTY"):
        self.model = model
        self.template = template
        self.use_api = use_api  # 新增：选择使用API还是PT引擎
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.debug_mode = False
        
        # 系统提示
        self.system_prompt = self._build_system_prompt()
        
        # 初始化PT引擎（如果不使用API模式）
        self.engine = None
        self.request_config = None
        if not self.use_api and SWIFT_AVAILABLE and self.model and self.template:
            try:
                logger.info(f"初始化PT引擎: {model}, 模板: {template}")
                
                # 如果model是字符串路径，需要先加载
                if isinstance(model, str):
                    model_obj, tokenizer = get_model_tokenizer(model)
                    template_obj = get_template(template, tokenizer)
                    self.engine = PtEngine.from_model_template(model_obj, template_obj, max_batch_size=1)
                else:
                    # 如果model已经是模型对象
                    self.engine = PtEngine.from_model_template(model, template, max_batch_size=1)
                
                self.request_config = RequestConfig(temperature=0.6, top_p=0.95, max_tokens=4096)
                logger.info("PT引擎初始化完成")
            except Exception as e:
                logger.error(f"PT引擎初始化失败: {e}")
                import traceback
                traceback.print_exc()
        elif not self.use_api:
            logger.warning("PT引擎模式：模型或模板未提供，或swift库不可用")
        
        # 初始化API客户端（如果使用API模式或debug模式）
        self.api_client = None
        self.api_model_name = None
        if self.use_api or self.debug_mode:
            try:
                self.api_client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base_url,
                    timeout=30
                )
                self.api_model_name = self.api_client.models.list().data[0].id
                if self.use_api:
                    logger.info(f"API客户端初始化完成，模型: {self.api_model_name}")
                else:
                    logger.info(f"Debug客户端初始化完成，模型: {self.api_model_name}")
            except Exception as e:
                logger.error(f"API客户端初始化失败: {e}")
                self.api_client = None
                self.api_model_name = None
        self.reawrd_prompt = self._build_reawrd_prompt()

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
    def _build_reawrd_prompt(self) -> str:
        """导入奖励提示"""
        with open("/data/dlf/code/Field-Fidelity/src/train/prompt/reward_process_atomic.md", "r") as f:
            return f.read()

        
    


    def prepare_messages(self, user_prompt,current_answer: str, reference_answer: str, question: str, original_images=None) -> List[Dict]:


        user_prompt = user_prompt.replace("{question}", question)
        user_prompt = user_prompt.replace("{reference_answer}", reference_answer)
        user_prompt = user_prompt.replace("{current_answer}", current_answer)

        # 如果有原始图片，则使用多模态格式
        if original_images and len(original_images) > 0:
            # 构建多模态消息
            user_content = []
            
           
            # 添加图片
            for img_info in original_images:
                try:
                    if isinstance(img_info, str):
                        # 字符串格式的图片路径
                        img_path = img_info
                    elif isinstance(img_info, dict):
                        # 字典格式的图片信息
                        img_path = img_info.get("path", img_info.get("url", ""))
                    else:
                        continue
                    
                    if img_path and os.path.exists(img_path):
                        # 读取图片并压缩
                        import base64
                        from PIL import Image
                        import io
                        
                        # 打开图片
                        with Image.open(img_path) as img:
                            # 计算当前像素数
                            current_pixels = img.width * img.height
                            max_pixels = 1003520
                            
                            # 如果超过最大像素限制，进行压缩
                            if current_pixels > max_pixels:
                                # 计算压缩比例
                                scale_factor = (max_pixels / current_pixels) ** 0.5
                                new_width = int(img.width * scale_factor)
                                new_height = int(img.height * scale_factor)
                                
                                # 调整图片大小
                                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                logger.info(f"图片压缩: {img_path} 从 {img.width}x{img.height} 压缩到 {new_width}x{new_height}")
                            
                            # 转换为RGB模式（如果需要）
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # 保存到内存缓冲区
                            buffer = io.BytesIO()
                            img.save(buffer, format='JPEG', quality=85, optimize=True)
                            image_data = buffer.getvalue()
                            buffer.close()
                            
                        # 转换为base64
                        base64_encoded = base64.b64encode(image_data).decode("utf-8")
                        
                        # 压缩后的图片统一使用JPEG格式
                        image_type = "image/jpeg"

                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_type};base64,{base64_encoded}"
                            }
                        })
                        logger.info(f"成功加载图片: {img_path}")
                    else:
                        logger.warning(f"图片路径不存在: {img_path}")
                        
                except Exception as e:
                    logger.error(f"处理图片失败: {e}")
                 # 先添加文本评估提示
            user_content.append({
                "type": "text",
                "text": user_prompt
            })
            
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content}
            ]
        else:
            # 纯文本消息
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

    def parse_result(self, judgment: str, question: str = "", current_answer: str = "", reference_answer: str = "", original_messages: Optional[List[Dict]] = None, original_images: Optional[List] = None) -> float:
        """解析评分结果并记录日志"""
        reward_logger.info("-" * 50)
        reward_logger.info("开始解析评估结果")
        reward_logger.info(f"问题: {question[:100]}...")
        reward_logger.info(f"模型回答: {current_answer[:100]}...")
        reward_logger.info(f"参考回答: {reference_answer[:100]}...")
        reward_logger.info(f"判定内容: {judgment}")
        
        # 记录原始图片信息
        if original_images:
            reward_logger.info(f"原始图片信息: {len(original_images)} 张图片")
            for i, img_info in enumerate(original_images):
                if isinstance(img_info, str):
                    reward_logger.info(f"  图片 {i+1}: {img_info}")
                elif isinstance(img_info, dict):
                    path = img_info.get('path', img_info.get('url', str(img_info)))
                    reward_logger.info(f"  图片 {i+1}: {path}")
        
        # 记录原始消息
        if original_messages:
            reward_logger.info("原始消息结构:")
            for i, msg in enumerate(original_messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if isinstance(content, str):
                    reward_logger.info(f"  消息 {i+1} - {role}: {content[:100]}...")
                elif isinstance(content, list):
                    reward_logger.info(f"  消息 {i+1} - {role}: [多模态内容, {len(content)} 个元素]")
                    for j, item in enumerate(content):
                        item_type = item.get('type', 'unknown')
                        if item_type == 'text':
                            text = item.get('text', '')[:50]
                            reward_logger.info(f"    元素 {j+1}: text - {text}...")
                        elif item_type == 'image_url':
                            image_url = item.get('image_url', {})
                            url = image_url.get('url', '')
                            if url.startswith('data:'):
                                reward_logger.info(f"    元素 {j+1}: image_url - [base64图片]")
                            else:
                                reward_logger.info(f"    元素 {j+1}: image_url - {url}")
                        else:
                            reward_logger.info(f"    元素 {j+1}: {item_type}")
        # 解析JSON格式的评分结果
        final_score = 0.5

        try:
            # 多种可能的评分表达方式
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
                score_match = re.search(pattern, judgment, re.IGNORECASE)
                if score_match:
                    try:
                        final_score = float(score_match.group(1))
                        score_found = True
                        break
                    except ValueError:
                        continue
            
            if not score_found:
                reward_logger.warning(f"judgment: {judgment}")
                reward_logger.warning("未找到评分，使用默认值0.5")
            
        except Exception as e:
            reward_logger.error(f"评分解析失败: {e}")
            final_score = 0.5

        # 确保分数在合理范围内
        final_score = max(min(final_score, 1.0), -1.0)

        record = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "current_answer": current_answer,
            "reference_answer": reference_answer,
            "judgment": judgment,
            "final_score": final_score,
            "original_messages": original_messages or [],
            "original_images": original_images or []
        }
        write_reward_record(REWARD_JSONL_PATH, record)
        
        reward_logger.info(f"最终奖励分数: {final_score:.3f}")
        reward_logger.info("-" * 50)
        
        return final_score

    def debug_compare_responses(self, eval_messages: List[Dict], sample_info: Dict) -> Dict:
        """Debug方法：对比本地引擎和localhost:8006的响应"""
        if not self.debug_mode:
            #logger.info(json.dumps(eval_messages, ensure_ascii=False, indent=2))
            return {"error": "Debug模式未启用"}
        
        debug_result = {
            "pt_engine_response": None,
            "api_response": None,
            "score_comparison": None,
            "message_structure": eval_messages
        }
        
        try:
            # 1. 测试PT引擎（如果可用）
            if not self.use_api and self.engine:
                logger.info("Debug: 测试PT引擎响应...")
                try:
                    pt_content = self.infer_with_engine(eval_messages)
                    pt_score = self.parse_result(
                        pt_content,
                        sample_info['question'],
                        sample_info['current_answer'],
                        sample_info['reference_answer'],
                        original_messages=sample_info['messages'],
                        original_images=sample_info['original_images']
                    )
                    debug_result["pt_engine_response"] = {
                        "content": pt_content,
                        "score": pt_score
                    }
                    logger.info(f"Debug: PT引擎 content: {pt_content[:100]}...")
                except Exception as e:
                    logger.error(f"Debug: PT引擎测试失败: {e}")
                    debug_result["pt_engine_response"] = {"error": str(e)}
            
            # 2. 测试API（如果可用）
            if self.api_client and self.api_model_name:
                logger.info("Debug: 测试API响应...")
                try:
                    api_content = self.infer_with_api(eval_messages)
                    api_score = self.parse_result(
                        api_content,
                        sample_info['question'],
                        sample_info['current_answer'],
                        sample_info['reference_answer'],
                        original_messages=sample_info['messages'],
                        original_images=sample_info['original_images']
                    )
                    debug_result["api_response"] = {
                        "content": api_content,
                        "score": api_score
                    }
                    logger.info(f"Debug: API content: {api_content[:100]}...")
                except Exception as e:
                    logger.error(f"Debug: API测试失败: {e}")
                    debug_result["api_response"] = {"error": str(e)}
            
            # 3. 记录message结构
            # logger.info("Debug: Message结构:")
            # logger.info(json.dumps(eval_messages, ensure_ascii=False, indent=2))
            
            return debug_result
            
        except Exception as e:
            logger.error(f"Debug对比失败: {e}")
            debug_result["error"] = str(e)
            return debug_result

    def __call__(self, inputs: List[Dict],**kwargs) -> torch.Tensor:
        """计算奖励分数"""
        trainer_state = kwargs.get('trainer_state')
        if trainer_state:
            global_step = getattr(trainer_state, 'global_step', 0)
            max_steps = getattr(trainer_state, 'max_steps', 0)
            logger.info(f"global_step: {global_step}, max_steps: {max_steps}")
        
        if not self.engine and not self.use_api:
            logger.error("PT引擎未初始化且未启用API模式")
            return torch.tensor([0.5] * len(inputs), dtype=torch.float32)
        
        if self.use_api and not self.api_client:
            logger.error("API模式已启用但API客户端未初始化")
            return torch.tensor([0.5] * len(inputs), dtype=torch.float32)
        
    
        assert inputs is not None, "inputs is None"
        logger.info(f"开始处理批次: {len(inputs)}样本")
        logger.info(f"推理模式: {'API' if self.use_api else 'PT引擎'}")
        reward_logger.info("=" * 60)
        reward_logger.info(f"开始新批次评估: {len(inputs)} 个样本")
        reward_logger.info(f"推理模式: {'API' if self.use_api else 'PT引擎'}")
        reward_logger.info("=" * 60)
        
        # 准备推理请求
        rm_inputs = []
        valid_indices = []
        sample_info = []  # 存储样本信息用于日志记录
        
        for idx, inp in enumerate(inputs):
            # 提取信息
            messages = inp.get('messages', [])
            current_answer = ""
            question = ""
            
            # 提取问题和回答
            if messages and len(messages) >= 2:
                for msg in messages:
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
            
            # 获取参考回答
            reference_answer = inp.get('reference') or inp.get('solution') or ''
            
            # 清理问题
            if "<image>" in question:
                question = question.replace("<image>", "")
            
            if current_answer and reference_answer:
                # 准备推理消息
                original_user_content = None
                # 提取原始用户消息内容（可能包含图片）
                for msg in messages:
                    if msg.get('role') == 'user':
                        original_user_content = msg.get('content')
                        break
                
                use_image_eval = False
                if use_image_eval:
                    original_images = inp.get('images', [])
                else:
                    original_images = []
                eval_messages = self.prepare_messages(self.reawrd_prompt, current_answer, reference_answer, question, original_images)
                rm_inputs.append(eval_messages)  # 直接存储消息，不包装成字典
                valid_indices.append(idx)
                
                # Debug模式：对比localhost:8006的响应
                if self.debug_mode and idx == 0:  # 只对第一个样本进行debug
                    debug_info = {
                        'question': question,
                        'current_answer': current_answer,
                        'reference_answer': reference_answer,
                        'messages': eval_messages,
                        'original_images': original_images
                    }
                    debug_result = self.debug_compare_responses(eval_messages, debug_info)
                    logger.info(f"Debug结果: {json.dumps(debug_result, ensure_ascii=False, indent=2)}")
                
                # 保留完整的原始消息，包括images信息
                complete_messages = messages.copy()
                
                # 保留原始images信息
                
                
                sample_info.append({
                    'question': question,
                    'current_answer': current_answer,
                    'reference_answer': reference_answer,
                    'messages': complete_messages,  # 原始消息
                    'original_images': original_images  # 原始图片信息
                })
                logger.info(f"样本 {idx + 1}: 准备评估")
                reward_logger.info(f"样本 {idx + 1}: 准备评估 - 问题长度: {len(question)}, 回答长度: {len(current_answer)}, 图片数: {len(original_images)}")
            else:
                logger.warning(f"样本 {idx + 1}: 缺少必要信息")
                reward_logger.warning(f"样本 {idx + 1}: 缺少必要信息 - current_answer: {bool(current_answer)}, reference_answer: {bool(reference_answer)}")
        
        # 批量推理
        rewards = [0.5] * len(inputs)  # 默认分数
        
        if rm_inputs:
            logger.info(f"开始批量推理: {len(rm_inputs)} 个请求")
            reward_logger.info(f"开始批量推理: {len(rm_inputs)} 个有效请求")
            
            # 根据模式选择推理方式
            if self.use_api:
                # API模式：逐个推理
                results = []
                for messages in rm_inputs:
                    try:
                        content = self.infer_with_api(messages)
                        # 模拟引擎返回格式
                        class MockResult:
                            def __init__(self, content):
                                self.choices = [MockChoice(content)]
                        
                        class MockChoice:
                            def __init__(self, content):
                                self.message = MockMessage(content)
                        
                        class MockMessage:
                            def __init__(self, content):
                                self.content = content
                        
                        results.append(MockResult(content))
                    except Exception as e:
                        logger.error(f"API推理失败: {e}")
                        results.append(None)
            else:
                # PT引擎模式：批量推理
                if self.engine is None:
                    raise RuntimeError("PT引擎未初始化")
                infer_requests = [InferRequest(messages=messages) for messages in rm_inputs]
                results = self.engine.infer(infer_requests, self.request_config, use_tqdm=False)
            
            # 解析结果
            for i, result in enumerate(results):
                try:
                    if result is None:
                        logger.warning(f"推理结果 {i+1}: 结果为None")
                        reward_logger.warning(f"推理结果 {i+1}: 结果为None")
                        continue
                        
                    # 安全访问result属性
                    choices = getattr(result, 'choices', None)
                    if choices and len(choices) > 0:
                        choice = choices[0]
                        message = getattr(choice, 'message', None)
                        if message:
                            content = getattr(message, 'content', '')
                            if isinstance(content, str) and content.strip():
                                # 传递样本信息给parse_result
                                info = sample_info[i]
                                score = self.parse_result(
                                    content, 
                                    info['question'], 
                                    info['current_answer'], 
                                    info['reference_answer'],
                                    original_messages=info['messages'], # 传递原始消息
                                    original_images=info['original_images'] # 传递原始图片信息
                                )
                                original_idx = valid_indices[i]
                                rewards[original_idx] = score
                                logger.info(f"样本 {original_idx + 1} 奖励: {score:.3f}")
                            else:
                                logger.warning(f"推理结果 {i+1}: 内容为空或格式异常")
                                reward_logger.warning(f"推理结果 {i+1}: 内容为空或格式异常")
                        else:
                            logger.warning(f"推理结果 {i+1}: 缺少message属性")
                            reward_logger.warning(f"推理结果 {i+1}: 缺少message属性")
                    else:
                        logger.warning(f"推理结果 {i+1}: 缺少choices属性")
                        reward_logger.warning(f"推理结果 {i+1}: 缺少choices属性")
                except Exception as e:
                    logger.error(f"处理推理结果 {i+1} 失败: {e}")
                    reward_logger.error(f"处理推理结果 {i+1} 失败: {e}")
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        logger.info(f"批次完成: 平均奖励 {avg_reward:.3f}")
        
        # 批次总结日志
        reward_logger.info("=" * 60)
        reward_logger.info(f"批次评估完成")
        reward_logger.info(f"总样本数: {len(inputs)}")
        reward_logger.info(f"有效评估数: {len(rm_inputs)}")
        reward_logger.info(f"平均奖励: {avg_reward:.3f}")
        reward_logger.info(f"奖励分布: 最小={min(rewards):.3f}, 最大={max(rewards):.3f}")
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
