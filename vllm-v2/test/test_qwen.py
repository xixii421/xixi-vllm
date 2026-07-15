# 确保安装了: pip install transformers torch
from transformers import AutoConfig, AutoModelForCausalLM

model_id = "Qwen/Qwen3-0.6B" 

print("1. 正在读取模型配置 (Config)...")
config = AutoConfig.from_pretrained(model_id)

print(f"隐藏层维度\t: {config.hidden_size}")
print(f"注意力头数\t: {config.num_attention_heads}")
print(f"KV头数    \t: {config.num_key_value_heads}")
print(f"层数      \t: {config.num_hidden_layers}")
print(f"词表大小  \t: {config.vocab_size}")
print("-" * 50)

print("2. 正在构建模型骨架...")
model = AutoModelForCausalLM.from_config(config)

print("3. 打印模型架构树:")
print(model)