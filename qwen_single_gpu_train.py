# # %%
# %cd /root/Code/DataX/datax/
# %load_ext autoreload
# %autoreload 2

# %%
"""
全量参数微调分数回归奖励模型
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class ContinuousRewardModel(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        # 加载基础模型（如Llama-2-7B）
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            # dtype=torch.float16,
            device_map=None,
            tie_word_embeddings=False,
            # gradient_checkpointing=True # 梯度checkpointing（减少显存）
        )
        self.config = self.base_model.config
        # 冻结基础模型（可选，若仅微调回归头）
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        
        # 添加回归头：输入基础模型的[CLS]隐藏状态，输出连续分数（0-1）
        self.regression_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 1),  # 基础模型输出维度→1维
            nn.Sigmoid()  # 激活函数，将输出限制在0-1之间
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # 前向传播：获取最后一层、最后一个token的隐藏状态（decoder-only模型的常用做法）
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True  # 必须开启，才能获取隐藏状态
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]
        # 回归头输出连续分数
        score = self.regression_head(last_hidden).squeeze(-1)  # [batch_size]
        return score

# %%
"""
Lora 微调奖励模型
"""
# from peft import LoraConfig, get_peft_model

# def build_lora_model(model_name_or_path: str) -> nn.Module:
#     # 1. 加载基础模型
#     base_model = AutoModel.from_pretrained(
#         model_name_or_path,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )
    
#     # 2. 配置LoRA
#     lora_config = LoraConfig(
#         r=8,  # LoRA秩，越小越节省显存
#         lora_alpha=32,  # 缩放因子
#         target_modules=["q_proj", "v_proj"],  # 仅微调注意力层的q/v投影
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM"  # 适配语言模型任务
#     )
    
#     # 3. 应用LoRA到基础模型
#     lora_model = get_peft_model(base_model, lora_config)
    
#     # 4. 添加回归头（同基础版）
#     class LoraRewardModel(nn.Module):
#         def __init__(self, lora_model):
#             super().__init__()
#             self.lora_model = lora_model
#             self.regression_head = nn.Sequential(
#                 nn.Linear(lora_model.config.hidden_size, 1),
#                 nn.Sigmoid()
#             )
        
#         def forward(self, input_ids, attention_mask=None):
#             outputs = self.lora_model(input_ids=input_ids, attention_mask=attention_mask)
#             cls_hidden = outputs.last_hidden_state[:, 0, :]
#             score = self.regression_head(cls_hidden).squeeze(-1)
#             return score
    
#     return LoraRewardModel(lora_model)

# %% [markdown]
# 下面是正式训练

# %%
system_prompt = """你是一个评论质量评估专家，需要根据以下标准评估评论的质量：
1. 相关性：评论与文章内容的关联程度，关联越紧密越好；
2. 情绪：优先奖励中性和正面情绪，打压负面或极端情绪内容；
3. 观点表达：有明显的个人观点倾向，观点清晰、逻辑越明确越好；
4. 询问求助：有意义、具体的提问优于无意义或宽泛问题；
5. 玩梗俏皮话：具有个人口吻、自然有趣的表达优于生硬或通用内容；
6. 信息量：具体、有实质内容的评论优于泛泛而谈、无营养内容；
7. 口语化：有明显的个人口吻特征，打压平铺直叙或僵硬的表达；
请输出0到1之间的连续分数，分数越高表示综合质量越好。"""

# %%
"""
数据处理环节

数据格式参考：
text,score
"这个手机壳的图案像我家猫，太可爱了！",0.92
"这个手机壳很好看",0.65
"这个手机壳的图案像猫咪，很可爱！",0.78
"这个手机壳的颜色很丑",0.21
"""
from datasets import load_dataset
from transformers import DataCollatorWithPadding

def prepare_dataset(model_name_or_path: str, data_path: str):
    # 1. 加载数据集
    dataset = load_dataset("csv", data_files=data_path)
    
    # 2. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token  # 大模型需设置pad_token
    # 关键调整：decoder-only模型需右边padding+用eos_token作为pad_token
    tokenizer.padding_side = "right"
    
    model_max_length = tokenizer.model_max_length
    # 3. 预处理函数：tokenize文本，保留连续分数标签
    def preprocess_function(examples):
        # Tokenize文本（max_length根据模型调整，如512）
        # 拼接System Prompt与每条评论（用ChatML格式增强模型理解）
        texts = []
        for text in examples["text"]:
            try:
                part0, part1 = text.split("<评论>", 1)  # 仅分割一次
                
                # 拼接ChatML格式（确保无嵌套）
                txt = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n<评论>{part1.strip()}\n{part0.strip()}<|im_end|>"
                )
                texts.append(txt)
            except Exception as e:
                print(f"警告：处理文本失败（{text}），错误：{e}")
                texts.append("")  # 填充空字符串，避免中断流程
        # Tokenize拼接后的文本
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=8192,  # 需覆盖System Prompt + 文本的长度
            padding="max_length",
            add_special_tokens=True
        )
        # 添加连续分数标签（转为float）
        tokenized["labels"] = [float(score) for score in examples["score"]]
        return tokenized
    
    # 4. 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    # 7. 验证数据格式（关键！确保无嵌套）
    sample = tokenized_dataset["train"][0]
    assert isinstance(sample["input_ids"], list), "input_ids必须是列表"
    assert isinstance(sample["input_ids"][0], int), "input_ids元素必须是整数"
    assert isinstance(sample["labels"], float), "labels必须是浮点数"
    # assert len(sample["input_ids"]) == model_max_length, "input_ids长度必须等于模型最大长度"
    
    # 5. 数据整理器（自动padding）
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    return tokenized_dataset, tokenizer, data_collator

# %%
"""
自定义RewardTrainer：适配连续回归
TRL 的RewardTrainer默认用对比损失（Pairwise），需重写compute_loss方法，改用回归损失（如 MSE）：
"""
from typing import Optional
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import Trainer, TrainingArguments

class ContinuousRewardTrainer(Trainer):
    
    def compute_loss(self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        重写损失计算：用MSE损失（Mean Squared Error）
        """
         # 1. 模型前向传播：输出连续分数
        # scores = model(
        #     input_ids=inputs["input_ids"],
        #     attention_mask=inputs["attention_mask"]
        # )
        # labels = inputs["labels"].to(scores.device)  # 确保设备一致  # 连续分数转为float
        # outputs = model(**inputs)
        # loss = F.mse_loss(outputs, labels)  # 回归任务用MSE损失
        # return (loss, outputs) if return_outputs else loss
    
        #取出连续分数labels（转为float）
        device = next(model.parameters()).device
        labels = inputs.pop("labels").to(device).float()
        # 前向传播
        outputs = model(**inputs)
        # MSE损失（回归任务）
        loss = torch.nn.functional.mse_loss(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, eval_pred):
        """
        自定义评估指标：回归任务常用MAE、RMSE、R²
        """
        predictions, labels = eval_pred
        predictions = predictions.squeeze()  # 从[batch_size,1]转为[batch_size]
        labels = labels.squeeze()
        
        # 计算指标
        mae = mean_absolute_error(labels, predictions)
        rmse = mean_absolute_error(labels, predictions, squared=False)
        r2 = r2_score(labels, predictions)
        
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

# %%
# output_dir = "./self_test/train/tensorboard"
# %reload_ext tensorboard
# %tensorboard --logdir '{output_dir}'/runs
# %tensorboard --logdir ./self_test/train/tensorboard/runs

# %%
# %env HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
# %env http_proxy=http://sys-proxy-rd-relay.byted.org:8118
# %env https_proxy=http://sys-proxy-rd-relay.byted.org:8118


import torch
# print("PyTorch的CUDA版本：", torch.version.cuda)  # 应返回正确版本（如11.8）
# print("PyTorch的CUDA版本：", torch.__version__)  # 
# print("CUDA可用吗？", torch.cuda.is_available())  # 应返回True
# print("当前GPU：", torch.cuda.current_device())  # 应返回0
# print("GPU名称：", torch.cuda.get_device_name(0))  # 应返回GPU型号（如NVIDIA RTX 3090）

# %%
"""
训练模型
"""
from transformers import TrainingArguments
import deepspeed

def train_continuous_reward_model():
    # 1. 配置参数
    model_name_or_path = "/mlx_devbox/users/wukunqiong/playground/arnold_workspace_root/Code/DataX/Qwen3-1.7B"  # 基础模型
    data_path = "/mlx_devbox/users/wukunqiong/playground/arnold_workspace_root/Code/DataX/datax/self_test/train/1031-thinking-优势分-训练1.csv"  # 连续分数数据
      # 模型保存路径
    output_dir = "./self_test/train/continuous_reward_model"
    
    # def model_builder():
        # 2. 构建模型（选LoRA或全量微调）
        # return ContinuousRewardModel(model_name_or_path)  # 全量微调
        # model = build_lora_model(model_name_or_path)  # LoRA微调
        
    model = ContinuousRewardModel(model_name_or_path)
    
    model = ContinuousRewardModel(model_name_or_path)
    # 检查两个张量是否共享内存（关闭后应为False）
    print(model.base_model.model.embed_tokens.weight.is_shared())  # 输出False
    print(model.base_model.lm_head.weight.is_shared())            # 输出False
    # 检查两个张量是否相同（关闭后应为不同）
    print(torch.allclose(model.base_model.model.embed_tokens.weight, model.base_model.lm_head.weight))  # 输出False
    
    # 3. 准备数据
    tokenized_dataset, tokenizer, data_collator = prepare_dataset(model_name_or_path, data_path)
    
    
    # import yaml  # 导入PyYAML库
    # 关键修正：用PyYAML加载YAML配置（替换原文件路径）
    # fsdp_config_path = "self_test/train/fsdp_config.yaml"
    # with open(fsdp_config_path, "r") as f:
    #     fsdp_config = yaml.safe_load(f)  # 将YAML转为Python字典

    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=1,
    #     gradient_accumulation_steps=8,
    #     num_train_epochs=4,
    #     fsdp_config=fsdp_config,
    #     fsdp="full_shard",
    #     weight_decay=0.01,
    #     save_strategy="epoch",
    #     # load_best_model_at_end=True,
    #     # push_to_hub=True,
    #     report_to="tensorboard"
    # )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # 批次大小，根据显存调整
        gradient_accumulation_steps=8,  # 梯度累积，模拟大批次
        learning_rate=1e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,  # 混合精度训练
        # report_to="swanlab"
        optim="paged_adamw_32bit",
        label_names=["labels"],
        report_to="tensorboard"
    )
    
    # 4. 初始化Trainer
    trainer = ContinuousRewardTrainer(
        model=model,
        train_dataset=tokenized_dataset['train'],
        # eval_dataset=tokenized_dataset.get("validation", None),  # 若有验证集
        args=training_args,
        data_collator=data_collator,
        processing_class=tokenizer
    )
    
    # 5. 启动训练
    trainer.train()
    
    # 6. 保存模型
    # trainer.save_model(output_dir)
    save_model_without_shared_memory(model, output_dir)
    tokenizer.save_pretrained(output_dir)
    
def save_model_without_shared_memory(model, save_path):
    # 分离嵌入层与输出层的共享张量
    if hasattr(model.base_model, 'lm_head') and hasattr(model.base_model.model, 'embed_tokens'):
        # 复制lm_head.weight，断开共享
        model.base_model.lm_head.weight = torch.nn.Parameter(
            model.base_model.lm_head.weight.detach().clone()
        )
    # 保存模型
    model.save_pretrained(save_path)

# 执行训练
train_continuous_reward_model()

# from accelerate import notebook_launcher
# from multiprocess import set_start_method
# if __name__ == "__main__":
#     set_start_method("spawn")
#     notebook_launcher(train_continuous_reward_model, args=(), num_processes=8)

# %%


# %%


# %% [markdown]
# 要在TRL库的`RewardTrainer`中添加**连续分数回归**的奖励打分头（即让奖励模型输出0-1的连续分数，而非Pairwise对比或分类结果），需**自定义模型结构**、**调整损失函数**和**适配数据格式**。以下是**完整的实现步骤+代码示例**，基于TRL v0.7.10（最新稳定版）和Hugging Face Transformers。
# 
# 
# ## **一、核心背景**
# TRL的`RewardTrainer`默认支持**Pairwise对比学习**（即输入两条文本，模型判断哪条更优），但**不直接支持连续回归任务**。要实现连续分数回归，需：
# 1. 将基础模型（如Llama-2）的**分类头替换为回归头**（输出连续值）；
# 2. 将训练损失从**交叉熵/对比损失**改为**回归损失**（如MSE、MAE）；
# 3. 调整数据格式：标签从“偏好标签”（如1/0表示哪条更优）改为**连续分数**（如0.85表示文本质量）。
# 
# 
# ## **二、具体实现步骤**
# ### **1. 环境准备**
# 安装依赖库（确保版本兼容）：
# ```bash
# pip install trl==0.7.10 transformers==4.34.0 datasets==2.14.5 peft==0.5.0 torch==2.0.1
# ```
# 
# 
# ### **2. 定义“带连续回归头的奖励模型”**
# 需**自定义模型结构**：基础模型（如Llama-2）+ 回归头（输出连续分数）。  
# 若需**参数高效微调**（如LoRA），可结合`peft`库。
# 
# 
# #### **2.1 基础版：全量微调的回归模型**
# ```python
# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer
# 
# class ContinuousRewardModel(nn.Module):
#     def __init__(self, model_name_or_path: str):
#         super().__init__()
#         # 加载基础模型（如Llama-2-7B）
#         self.base_model = AutoModel.from_pretrained(
#             model_name_or_path,
#             torch_dtype=torch.float16,  # 用16位浮点节省显存
#             device_map="auto"
#         )
#         # 冻结基础模型（可选，若仅微调回归头）
#         # for param in self.base_model.parameters():
#         #     param.requires_grad = False
#         
#         # 添加回归头：输入基础模型的[CLS]隐藏状态，输出连续分数（0-1）
#         self.regression_head = nn.Sequential(
#             nn.Linear(self.base_model.config.hidden_size, 1),  # 基础模型输出维度→1维
#             nn.Sigmoid()  # 激活函数，将输出限制在0-1之间
#         )
# 
#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
#         # 基础模型前向传播：取[CLS] token的隐藏状态（最后一层）
#         outputs = self.base_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         cls_hidden = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
#         
#         # 回归头输出连续分数
#         score = self.regression_head(cls_hidden).squeeze(-1)  # [batch_size]
#         return score
# ```
# 
# 
# #### **2.2 进阶版：LoRA参数高效微调（推荐大模型）**
# 若基础模型是7B/13B等大模型，全量微调显存不足，需用**LoRA**（Low-Rank Adaptation）仅微调部分参数：
# ```python
# from peft import LoraConfig, get_peft_model
# 
# def build_lora_model(model_name_or_path: str) -> nn.Module:
#     # 1. 加载基础模型
#     base_model = AutoModel.from_pretrained(
#         model_name_or_path,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )
#     
#     # 2. 配置LoRA
#     lora_config = LoraConfig(
#         r=8,  # LoRA秩，越小越节省显存
#         lora_alpha=32,  # 缩放因子
#         target_modules=["q_proj", "v_proj"],  # 仅微调注意力层的q/v投影
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM"  # 适配语言模型任务
#     )
#     
#     # 3. 应用LoRA到基础模型
#     lora_model = get_peft_model(base_model, lora_config)
#     
#     # 4. 添加回归头（同基础版）
#     class LoraRewardModel(nn.Module):
#         def __init__(self, lora_model):
#             super().__init__()
#             self.lora_model = lora_model
#             self.regression_head = nn.Sequential(
#                 nn.Linear(lora_model.config.hidden_size, 1),
#                 nn.Sigmoid()
#             )
#         
#         def forward(self, input_ids, attention_mask=None):
#             outputs = self.lora_model(input_ids=input_ids, attention_mask=attention_mask)
#             cls_hidden = outputs.last_hidden_state[:, 0, :]
#             score = self.regression_head(cls_hidden).squeeze(-1)
#             return score
#     
#     return LoraRewardModel(lora_model)
# ```
# 
# 
# ### **3. 准备“连续分数”训练数据**
# 需将数据处理为**“文本→连续分数”**的格式（标签是0-1的float值）。例如，用`datasets`库加载CSV数据：
# 
# 
# #### **3.1 数据格式示例（CSV文件）**
# ```csv
# text,score
# "这个手机壳的图案像我家猫，太可爱了！",0.92
# "这个手机壳很好看",0.65
# "这个手机壳的图案像猫咪，很可爱！",0.78
# "这个手机壳的颜色很丑",0.21
# ```
# 
# 
# #### **3.2 数据加载与预处理**
# ```python
# from datasets import load_dataset
# from transformers import DataCollatorWithPadding
# 
# def prepare_dataset(model_name_or_path: str, data_path: str):
#     # 1. 加载数据集
#     dataset = load_dataset("csv", data_files=data_path)
#     
#     # 2. 加载tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#     tokenizer.pad_token = tokenizer.eos_token  # 大模型需设置pad_token
#     
#     # 3. 预处理函数：tokenize文本，保留连续分数标签
#     def preprocess_function(examples):
#         # Tokenize文本（max_length根据模型调整，如512）
#         tokenized = tokenizer(
#             examples["text"],
#             truncation=True,
#             max_length=512,
#             padding="max_length"
#         )
#         # 添加连续分数标签（转为float）
#         tokenized["labels"] = [float(score) for score in examples["score"]]
#         return tokenized
#     
#     # 4. 应用预处理
#     tokenized_dataset = dataset.map(preprocess_function, batched=True)
#     
#     # 5. 数据整理器（自动padding）
#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#     
#     return tokenized_dataset, tokenizer, data_collator
# ```
# 
# 
# ### **4. 自定义`RewardTrainer`：适配连续回归**
# TRL的`RewardTrainer`默认用**对比损失**（Pairwise），需重写`compute_loss`方法，改用**回归损失**（如MSE）：
# ```python
# from trl import RewardTrainer
# import torch.nn.functional as F
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 
# class ContinuousRewardTrainer(RewardTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         重写损失计算：用MSE损失（Mean Squared Error）
#         """
#         # 1. 模型前向传播：输出连续分数
#         scores = model(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"]
#         )
#         
#         # 2. 获取真实标签（连续分数）
#         labels = inputs["labels"].to(scores.device)  # 确保设备一致
#         
#         # 3. 计算回归损失（MSE）
#         loss = F.mse_loss(scores, labels)
#         
#         # 4. 返回损失（可选返回输出）
#         return (loss, scores) if return_outputs else loss
# 
#     def compute_metrics(self, eval_pred):
#         """
#         自定义评估指标：回归任务常用MAE、RMSE、R²
#         """
#         predictions, labels = eval_pred
#         predictions = predictions.squeeze()  # 从[batch_size,1]转为[batch_size]
#         labels = labels.squeeze()
#         
#         # 计算指标
#         mae = mean_absolute_error(labels, predictions)
#         rmse = mean_squared_error(labels, predictions, squared=False)
#         r2 = r2_score(labels, predictions)
#         
#         return {
#             "mae": mae,
#             "rmse": rmse,
#             "r2": r2
#         }
# ```
# 
# 
# ### **5. 训练模型**
# 整合以上组件，启动训练：
# ```python
# def train_continuous_reward_model():
#     # 1. 配置参数
#     model_name_or_path = "meta-llama/Llama-2-7b-hf"  # 基础模型
#     data_path = "reward_data.csv"  # 连续分数数据
#     output_dir = "continuous_reward_model"  # 模型保存路径
#     
#     # 2. 构建模型（选LoRA或全量微调）
#     # model = ContinuousRewardModel(model_name_or_path)  # 全量微调
#     model = build_lora_model(model_name_or_path)  # LoRA微调
#     
#     # 3. 准备数据
#     tokenized_dataset, tokenizer, data_collator = prepare_dataset(model_name_or_path, data_path)
#     
#     # 4. 初始化Trainer
#     trainer = ContinuousRewardTrainer(
#         model=model,
#         train_dataset=tokenized_dataset["train"],
#         eval_dataset=tokenized_dataset.get("validation", None),  # 若有验证集
#         args=TrainingArguments(
#             output_dir=output_dir,
#             per_device_train_batch_size=4,  # 批次大小，根据显存调整
#             gradient_accumulation_steps=4,  # 梯度累积，模拟大批次
#             learning_rate=1e-4,
#             num_train_epochs=3,
#             logging_steps=10,
#             save_strategy="epoch",
#             fp16=True,  # 混合精度训练
#             report_to="tensorboard"
#         ),
#         data_collator=data_collator,
#         tokenizer=tokenizer
#     )
#     
#     # 5. 启动训练
#     trainer.train()
#     
#     # 6. 保存模型
#     trainer.save_model(output_dir)
#     tokenizer.save_pretrained(output_dir)
# 
# # 执行训练
# train_continuous_reward_model()
# ```
# 
# 
# ## **三、关键细节说明**
# ### **1. 回归头的激活函数**
# 若连续分数需限制在**0-1之间**（如“质量分数”），回归头需加`nn.Sigmoid()`激活；若分数范围无限制（如“情感极性”），可省略激活函数。
# 
# 
# ### **2. 损失函数选择**
# - **MSE（Mean Squared Error）**：对异常值敏感，适合标签分布较集中的场景；
# - **MAE（Mean Absolute Error）**：对异常值鲁棒，适合标签分布较分散的场景；
# - **Huber Loss**：结合MSE和MAE的优点，小误差用MSE，大误差用MAE，推荐用于噪声数据。
# 
# 
# ### **3. 评估指标**
# 连续回归任务的核心指标：
# - **MAE**：平均绝对误差（越小越好）；
# - **RMSE**：均方根误差（越小越好，对大误差更敏感）；
# - **R²**：决定系数（越接近1越好，表示模型解释方差的能力）。
# 
# 
# ### **4. 大模型显存优化**
# - 用**16位浮点（torch.float16）**加载模型，显存占用减半；
# - 用**LoRA**仅微调注意力层，显存占用可从40GB+降至8GB+（7B模型）；
# - 用**梯度累积**（`gradient_accumulation_steps`）模拟大批次，提升训练稳定性。
# 
# 
# ## **四、验证模型效果**
# 训练完成后，加载模型并测试连续分数预测：
# ```python
# def predict_score(model_path: str, text: str):
#     # 加载模型和tokenizer
#     model = ContinuousRewardModel.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     
#     # Tokenize输入
#     inputs = tokenizer(
#         text,
#         truncation=True,
#         max_length=512,
#         padding="max_length",
#         return_tensors="pt"
#     )
#     
#     # 模型推理
#     model.eval()
#     with torch.no_grad():
#         score = model(**inputs).item()
#     
#     print(f"文本：{text}")
#     print(f"预测分数：{score:.2f}")
# 
# # 测试示例
# predict_score("continuous_reward_model", "这个手机壳的图案像我家小主子，太萌了！")
# ```
# 
# 
# ## **总结**
# TRL的`RewardTrainer`实现连续分数回归的核心是：
# 1. **自定义回归模型**：将分类头改为回归头；
# 2. **调整损失函数**：从对比损失改为回归损失；
# 3. **适配数据格式**：标签从“偏好”改为连续分数。
# 
# 这种方法适用于**需要连续质量评估**的场景（如“评论质量分数”“内容相关性分数”），是Pairwise对比学习的补充。


