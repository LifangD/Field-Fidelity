# IDK Evaluation Framework

基于 lmms-eval 框架实现的 IDK (I Don't Know) 评估功能，用于测量视觉语言模型在不确定情况下的表达能力。

## 功能特性

### IDK 指标
- **IDK Accuracy**: 模型是否正确表达了不确定性
- **VQA Substring Accuracy**: 基于子串匹配的VQA准确率（Cha et al., 2024）
- **Precision/Recall/F1**: IDK检测的精确率、召回率和F1分数

### IDK 关键词
基于论文 Table 5 的关键词列表：
```
ambiguous, bad question, cannot confirm, depend, don't know, 
it is difficult, i can't, none, not clear, not sure, sorry, 
hard to determine, not possible, uncertain, unanswerable, unknown
```

## 安装和设置

### 1. 环境准备
```bash
cd /data/dlf/code/Field-Fidelity/tools/lmms-eval
pip install -e .
```

### 2. 数据准备
确保 VQAv2-IDK 数据集可用：
- 训练集: `/data/dlf/code/Field-Fidelity/data/idk/data/VQAv2-IDK-train.json`
- 验证集: `/data/dlf/code/Field-Fidelity/data/idk/data/VQAv2-IDK-val.json`

## 使用方法

### 1. 测试IDK任务实现
```bash
python /data/dlf/code/Field-Fidelity/scripts/test_idk_task.py
```

### 2. 运行IDK评估
```bash
# 基本用法
python /data/dlf/code/Field-Fidelity/scripts/run_idk_evaluation.py \
    --model llava_hf \
    --model_args "pretrained=liuhaotian/llava-v1.5-7b" \
    --tasks idk_vqav2_val

# 测试模式（只评估10个样本）
python /data/dlf/code/Field-Fidelity/scripts/run_idk_evaluation.py \
    --model llava_hf \
    --model_args "pretrained=liuhaotian/llava-v1.5-7b" \
    --tasks idk_vqav2_val \
    --test_mode

# 使用lmms-eval直接运行
cd /data/dlf/code/Field-Fidelity/tools/lmms-eval
python -m lmms_eval \
    --model llava_hf \
    --model_args "pretrained=liuhaotian/llava-v1.5-7b" \
    --tasks idk_vqav2_val \
    --batch_size 1 \
    --log_samples \
    --output_path ./outputs/idk_results
```

### 3. 支持的任务
- `idk_vqav2_val`: IDK VQAv2 验证集评估
- `idk_vqav2_test`: IDK VQAv2 测试集评估（生成提交文件）

### 4. 支持的模型
所有 lmms-eval 支持的模型都可以使用，例如：
- `llava_hf`: LLaVA 模型
- `qwen2_vl_hf`: Qwen2-VL 模型  
- `instructblip_hf`: InstructBLIP 模型
- 等等...

## 输出结果

### 评估指标
```json
{
  "idk_accuracy": 0.75,           // IDK准确率
  "vqa_substring_accuracy": 0.68, // VQA子串匹配准确率
  "idk_precision": 0.72,          // IDK精确率
  "idk_recall": 0.78,             // IDK召回率  
  "idk_f1": 0.75                  // IDK F1分数
}
```

### 详细日志
- 每个样本的预测结果
- 匹配的IDK关键词
- 混淆矩阵统计
- 提交文件（测试集）

## 文件结构

```
Field-Fidelity/
├── tools/lmms-eval/lmms_eval/tasks/idk_vqav2/
│   ├── __init__.py
│   ├── _default_template_idk_vqav2_yaml      # 默认配置模板
│   ├── idk_vqav2_val.yaml                   # 验证集配置
│   ├── idk_vqav2_test.yaml                  # 测试集配置
│   └── utils.py                             # 核心实现
├── scripts/
│   ├── test_idk_task.py                     # 功能测试脚本
│   └── run_idk_evaluation.py                # 评估运行脚本
└── outputs/
    └── idk_evaluation/                      # 评估结果输出
```

## 实现细节

### IDK 检测逻辑
1. **关键词匹配**: 检查回答中是否包含预定义的IDK关键词
2. **不确定性判断**: 根据真实答案判断是否应该表达不确定性
3. **准确率计算**: 比较模型表达与期望表达的一致性

### VQA 子串匹配
基于 Cha et al., 2024 的方法：
- 如果真实答案是预测句子的子串，则认为正确
- 不区分大小写
- 支持多个真实答案

### 聚合函数
- **精确率**: TP / (TP + FP)
- **召回率**: TP / (TP + FN)  
- **F1分数**: 2 * Precision * Recall / (Precision + Recall)

## 扩展支持

### 添加新的IDK任务
1. 创建新的任务目录：`lmms_eval/tasks/idk_<dataset>/`
2. 实现数据加载和处理函数
3. 配置YAML文件
4. 复用IDK关键词和指标计算逻辑

### 自定义IDK关键词
在 `utils.py` 中修改 `IDKKeywords.KEYWORDS` 列表

## 故障排除

### 常见问题
1. **数据集路径错误**: 确保数据集文件存在且路径正确
2. **模型加载失败**: 检查模型名称和参数是否正确
3. **内存不足**: 减少batch_size或使用limit参数

### 调试模式
```bash
# 启用详细日志
export LMMS_EVAL_DEBUG=1

# 只评估少量样本
python run_idk_evaluation.py --test_mode --limit 5
```

## 参考文献

- Cha et al., 2024: VQA evaluation methodology
- lmms-eval framework: https://github.com/EvolvingLMMs-Lab/lmms-eval
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
