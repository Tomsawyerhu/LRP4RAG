import jsonlines
import torch
from transformers import AutoTokenizer
from lxt.models.qwen2 import Qwen2ForCausalLM, attnlrp
from transformers import Qwen2Config
from lxt.utils import pdf_heatmap, clean_tokens

path = "/mnt/data/hhc/Qwen2.5-3B"

# 加载配置并设置参数
config = Qwen2Config.from_pretrained(path)
config._attn_implementation = 'eager'
config.torch_dtype = torch.float16

# 加载模型和 tokenizer
model = Qwen2ForCausalLM.from_pretrained(path, config=config, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(path)
# print(tokenizer.eos_token)
# 启用梯度检查点以节省内存
model.gradient_checkpointing_enable()
# 应用 AttnLRP 规则
attnlrp.register(model)


def lrp(prompt, max_token=1024):
    # 获取输入嵌入以便计算梯度
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    input_embeds = model.get_input_embeddings()(input_ids)

    # 初始化输出序列
    generated_tokens = []
    generated_token_ids = []
    relevance_scores = []

    # 生成直到遇到终止符
    while len(generated_tokens) < max_token:
        # 前向传播
        output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
        max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)

        # 反向传播计算梯度
        max_logits.backward(max_logits)
        relevance = input_embeds.grad.float().sum(-1).cpu()[0].tolist()
        relevance_scores.append(relevance)

        # 获取下一个 token
        next_token_id = max_indices.item()
        generated_token_ids.append(next_token_id)
        next_token = tokenizer.decode(next_token_id)
        next_token_id = torch.tensor([[next_token_id]])
        next_token_id = next_token_id.to(model.device)
        print(next_token, end=" ")
        generated_tokens.append(next_token)

        # 检查是否达到终止符
        if next_token == tokenizer.eos_token or next_token == "":
            break

        input_ids = torch.cat((input_ids, next_token_id), dim=-1)
        input_ids = input_ids.to(model.device)
        input_embeds = model.get_input_embeddings()(input_ids)
    return generated_tokens, generated_token_ids, relevance_scores


def filter_special_tokens(tokens: list):
    # 过滤掉特殊 token
    special_tokens = [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token, tokenizer.sep_token,
                      tokenizer.cls_token]
    filtered_tokens = [token for token in tokens if token not in special_tokens]

    # 将 tokens 重新组合成文本
    filtered_text = tokenizer.convert_tokens_to_string(filtered_tokens)
    return filtered_text


if __name__ == '__main__':
    input_file = "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_filtered.jsonl"
    output_file = "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_output.jsonl"
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, 'w') as writer:
        for record in reader:
            context = record["context"]
            instruction = record["instruction"]
            template = {
                "role": "user",
                "content": context + "Read the above context and answer the following question.\n" + instruction
            }
            # print(context)
            # prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True
            prompt = context + "Read the above context and answer the following question.\n" + instruction
            generated_tokens, generated_token_ids, relevance_scores = lrp(prompt)
            # 看回答有没有被截断
            if generated_tokens[-1] != tokenizer.eos_token:
                continue
            generated_text = filter_special_tokens(generated_tokens)
            print(generated_text)
            print("------------------------------")
            record["prompt"] = prompt
            record["answer"] = generated_text
            record["generated_tokens"] = generated_tokens
            record["generated_token_ids"] = generated_token_ids
            record["relevance_scores"] = relevance_scores
            record["model"] = "Qwen2.5-3B"
            writer.write(record)
