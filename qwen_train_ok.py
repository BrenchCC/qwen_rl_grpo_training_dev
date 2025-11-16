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
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,PreTrainedModel, Qwen3Config,Qwen3ForCausalLM,TrainerCallback,
    Qwen2Config,Qwen2ForCausalLM,Qwen2PreTrainedModel,Qwen2Model
)


class ContinuousRewardModel(Qwen2PreTrainedModel):
    
    # 关键：指定config类（与基础模型一致，如Qwen3Config）
    config_class = Qwen2Config  
    # 关键：指定模型类型（自定义，用于识别）
    model_type = "continuous_reward_model" 
    # is_parallelizable = True
    
    def __init__(self, config):
        super().__init__(config)
        # 加载基础模型（如Llama-2-7B）
        self.model = Qwen2Model(
            config
        )

        # 启用gradient_checkpointing（关键：节省显存）
        # self.model.gradient_checkpointing_enable()
        # 2. 关闭权重共享（避免保存错误）
        # self.model.tie_word_embeddings = False
        # 冻结基础模型（可选，若仅微调回归头）
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        
        # 添加回归头：输入基础模型的[CLS]隐藏状态，输出连续分数（0-1）
        self.regression_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 1),  # 基础模型输出维度→1维
            nn.Sigmoid()  # 激活函数，将输出限制在0-1之间
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # 前向传播：获取最后一层、最后一个token的隐藏状态（decoder-only模型的常用做法）
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 必须开启，才能获取隐藏状态
            use_cache=False  # 关键：训练时必须关闭cache
        )

        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]
         # ✅ 关键修复2：清理中间变量
        del outputs
        # 回归头输出连续分数
        score = self.regression_head(last_hidden).squeeze(-1)  # [batch_size]

        return score


# 下面是正式训练

# %%
system_prompt = """你是评论质量评估专家，需评估以下维度并输出0-1分数（越高越好）：
1. 相关性：与内容关联紧密；
2. 情绪：中性/正面优于负面；
3. 观点：清晰有逻辑；
4. 提问：具体有意义；
5. 俏皮话：自然有趣。"""

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

        # max_length = max([len(x) for x in examples["text"]])
        
        # Tokenize拼接后的文本
        tokenized = tokenizer(
            texts,
            truncation=True,
            # max_length=6000,  # 需覆盖System Prompt + 文本的长度
            padding="do_not_pad",
            # padding="max_length",
            add_special_tokens=True
        )
        # 添加连续分数标签（转为float）
        tokenized["labels"] = [float(score) for score in examples["score"]]
        return tokenized
    
    # 4. 应用预处理
    tokenized_dataset = dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=dataset["train"].column_names,
        # cache_file_name="./tokenized_imdb.cache"  # 缓存文件路径（CPU磁盘）
    )

    # tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.05, random_state=42)

    # 关键优化2：自定义DataCollator，按batch内最大长度padding
    class DynamicPaddingCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            
        def __call__(self, features):

            # 1. 提取batch内的input_ids、attention_mask、labels
            input_ids = [f["input_ids"] for f in features]
            attention_mask = [f["attention_mask"] for f in features]
            labels = [float(f["labels"]) for f in features]  # 确保label是float
            
            # 2. 计算batch内的最大长度（避免固定max_length）
            max_length = max(len(ids) for ids in input_ids)
            # print(f"max_length==={max_length}")
            
            # 3. 使用tokenizer.pad()批量padding（C++优化，效率更高）
            padded_batch = self.tokenizer.pad(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"  # 直接返回Tensor，避免手动转换
            )
            
            # 4. 添加labels（确保设备与模型一致）
            padded_batch["labels"] = torch.tensor(labels, dtype=torch.bfloat16)
            
            return padded_batch

            # # 找出batch内最大长度
            # max_length = max(len(f["input_ids"]) for f in features)
            
            # batch = {
            #     "input_ids": [],
            #     "attention_mask": [],
            #     "labels": []
            # }
            
            # for f in features:
            #     input_ids = f["input_ids"]
            #     attention_mask = f["attention_mask"]
                
            #     # 动态padding到batch最大长度
            #     padding_length = max_length - len(input_ids)
            #     input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            #     attention_mask = attention_mask + [0] * padding_length
                
            #     batch["input_ids"].append(input_ids)
            #     batch["attention_mask"].append(attention_mask)
            #     batch["labels"].append(f["labels"])
            
            # # 转为tensor
            # batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.long)
            # batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.long)
            # batch["labels"] = torch.tensor(batch["labels"], dtype=torch.float)
            
            # return batch

    data_collator = DynamicPaddingCollator(tokenizer)


    # 5. 数据整理器（自动padding）
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
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
from deepspeed.accelerator import get_accelerator

class EmptyCacheCallback(TrainerCallback):
    def __init__(self, every_n_steps=10):
        self.every_n_steps = every_n_steps  # 每10步清理一次

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            get_accelerator().empty_cache()  # 所有GPU同步清理

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

        # ✅ 关键修复5：训练过程中定期清理
        # if self.state.global_step % 30 == 0:
        #     get_accelerator().empty_cache()

        return (loss, outputs) if return_outputs else loss

    # def compute_metrics(self, eval_pred):
    #     """
    #     自定义评估指标：回归任务常用MAE、RMSE、R²
    #     """
    #     predictions, labels = eval_pred
    #     predictions = predictions.squeeze()  # 从[batch_size,1]转为[batch_size]
    #     labels = labels.squeeze()
        
    #     # 计算指标
    #     mae = mean_absolute_error(labels, predictions)
    #     rmse = mean_absolute_error(labels, predictions, squared=False)
    #     r2 = r2_score(labels, predictions)
        
    #     return {
    #         "mae": mae,
    #         "rmse": rmse,
    #         "r2": r2
    #     }

"""
训练模型
"""
from transformers import TrainingArguments
import deepspeed

def train_continuous_reward_model():
    # 1. 配置参数
    model_name_or_path = "/mnt/bn/wkq-mox/models/Qwen2.5-1.5B"  # 基础模型
    data_path = "/mlx_devbox/users/wukunqiong/playground/arnold_workspace_root/Code/Qwen_ar_score/train/1031-thinking-优势分-训练2.csv"  # 连续分数数据
      # 模型保存路径
    output_dir = "/mnt/bn/wkq-mox/continuous_reward_model-1107"
    
    # def model_builder():
        # 2. 构建模型（选LoRA或全量微调）
        # return ContinuousRewardModel(model_name_or_path)  # 全量微调
        # model = build_lora_model(model_name_or_path)  # LoRA微调
        
    # 加载模型配置
    config = Qwen2Config.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)

    # 关键优化3：启用Flash Attention 2（如果支持）
    # try:
    #     config._attn_implementation = "flash_attention_2"
    #     print("启用Flash Attention 2")
    # except:
    #     print("Flash Attention 2不可用，使用默认attention")
    
    # 创建模型（不要直接from_pretrained，让DeepSpeed处理）
    model = ContinuousRewardModel(config)
    
    # model = ContinuousRewardModel.from_pretrained(
    #     model_name_or_path,
    #     dtype=torch.float16,
    #     device_map=None
    # )

    # 加载预训练权重到model.model
    pretrained_state = Qwen2ForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16  # 改用bfloat16，比fp16更稳定
    ).state_dict()
    
    # 只加载base model的权重
    model.model.load_state_dict(pretrained_state, strict=False)

    # 3. 准备数据
    tokenized_dataset, tokenizer, data_collator = prepare_dataset(model_name_or_path, data_path)

    # import os
    # # 1. 启用CPU offload for activations
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,  # 保持为1
        gradient_accumulation_steps=8,  # 增加到16，减少更新频率
        learning_rate=1e-5,  # 稍微提高学习率以补偿更大的累积步数
        num_train_epochs=5,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="epoch",
        # save_strategy="steps",

        # ✅ 关键修复4：定期清理显存
        logging_steps=2,
        # save_steps=20,
        # eval_steps=20,


        fp16=False,  # 改用bf16代替fp16
        bf16=False,
        # bf16_full_eval=True,
        # gradient_checkpointing=True,  # 关键：启用gradient checkpointing
        optim="paged_adamw_8bit",  # DeepSpeed会覆盖这个配置
        label_names=["labels"],
        report_to="tensorboard",
        deepspeed="/mlx_devbox/users/wukunqiong/playground/arnold_workspace_root/Code/Qwen_ar_score/train/ds_config.json",
        # max_grad_norm=1.0,  # 添加梯度裁剪
        # 数据加载器配置（关键）
        dataloader_persistent_workers=True,  # 保持线程，CPU缓存Batch
        dataloader_pin_memory=True,          # 加速向GPU传输（CPU固定内存）
        dataloader_num_workers=4,            # 数据加载线程数

        # evaluation_strategy="epoch",  # 每个epoch结束后用验证集评估
        metric_for_best_model="accuracy",  # 用“准确率”判断最优模型
    )

    # from deepspeed.accelerator import get_accelerator
    # class EmptyCacheCallback(TrainerCallback):
    #     def on_epoch_end(self, args, state, control, **kwargs):
    #         get_accelerator().empty_cache()  # 所有rank同步清理缓存
    
    # 4. 初始化Trainer
    trainer = ContinuousRewardTrainer(
        model=model,
        train_dataset=tokenized_dataset['train'],
        # eval_dataset=tokenized_dataset['test'],
        # eval_dataset=tokenized_dataset.get("validation", None),  # 若有验证集
        args=training_args,
        data_collator=data_collator,
        processing_class=tokenizer,
        # callbacks=[EmptyCacheCallback()],
    )

    
    # 找到最新的checkpoint
    latest_checkpoint = find_latest_checkpoint(training_args.output_dir)

    # 训练前清理显存
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    if latest_checkpoint is None:
        # 5. 启动训练
        trainer.train()
    else:
        print(f"从{latest_checkpoint}启动训练")
        # 5. 启动训练（自动从latest_checkpoint恢复，若无则从头开始）
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    
    # 6. 保存模型
    trainer.save_model(f"{output_dir}_final")
    model.save_pretrained(f"{output_dir}_final_by_model")
    # save_model_without_shared_memory(model, output_dir)
    # tokenizer.save_pretrained(output_dir)
    


import os

def find_latest_checkpoint(output_dir):
    """找到output_dir下最新的checkpoint（步数最大的）"""
    if not os.path.exists(output_dir):
        return None  # 无checkpoint，从头开始
    
    # 列出所有checkpoint文件夹（格式：checkpoint-{step}）
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None  # 无checkpoint，从头开始
    
    # 按步数排序，取最大的
    latest_checkpoint = max(
        checkpoints,
        key=lambda x: int(x.split("-")[-1])  # 提取step数（如"checkpoint-200"→200）
    )
    return os.path.join(output_dir, latest_checkpoint)


os.environ["MASTER_PORT"] = "29501"  # 换一个未被占用的端口
# 训练前清理显存
import gc
gc.collect()
torch.cuda.empty_cache()
# 执行训练
train_continuous_reward_model()



# from accelerate import notebook_launcher
# from multiprocess import set_start_method
# if __name__ == "__main__":
#     set_start_method("spawn")
#     notebook_launcher(train_continuous_reward_model, args=(), num_processes=8)

# %%
