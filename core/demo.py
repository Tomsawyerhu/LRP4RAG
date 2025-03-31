import torch
from transformers import AutoTokenizer
from lxt.models.qwen2 import Qwen2ForCausalLM, attnlrp
from transformers import Qwen2Config
from lxt.utils import pdf_heatmap, clean_tokens

path = "/mnt/data/hhc/Qwen2.5-Coder-0.5B"

# 加载配置并设置参数
config = Qwen2Config.from_pretrained(path)
config._attn_implementation = 'eager'
config.torch_dtype = torch.float16

# 加载模型和 tokenizer
model = Qwen2ForCausalLM.from_pretrained(path, config=config, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(path)

# 启用梯度检查点以节省内存
model.gradient_checkpointing_enable()

# 应用 AttnLRP 规则
attnlrp.register(model)

# 输入文本
# prompt = """\
# Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
# Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""

prompt="""
The city of Elara has two main parks: Green Park and Blue Park. Green Park is known for its large lake, which is a popular spot for boating and fishing. The lake in Green Park is maintained by the Elara Water Management Department. Blue Park, on the other hand, is famous for its extensive flower gardens and walking trails.
The Elara Environmental Protection Agency (EPA) regularly monitors the water quality of the lake in Green Park. They have recently identified an increase in algae blooms, which can be harmful to the ecosystem. The EPA is working with the Elara Water Management Department to address this issue.
Blue Park has also been facing some environmental challenges. The flower gardens are being affected by a new type of pest, and the Elara Parks and Recreation Department is working on a solution. The Elara EPA is providing support by monitoring the impact of the pests on the local wildlife.
The Elara Water Management Department is responsible for maintaining the water infrastructure throughout the city, including the lake in Green Park. They have implemented a new water treatment system to reduce the algae blooms. Additionally, they are working with the Elara Parks and Recreation Department to ensure that the water in the fountains and ponds in both parks remains clean.
Which department is responsible for addressing the algae blooms in the lake in Green Park?
"""

# 获取输入嵌入以便计算梯度
input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(input_ids)
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
tokens = clean_tokens(tokens)

# 初始化输出序列
generated_tokens = []
relevance_scores = []

# 生成直到遇到终止符
while True:
    # 前向传播
    output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
    # max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)

    k = 2
    max_logits, max_indices = torch.topk(output_logits[0, -1, :], k=k, dim=-1)
    relevance_current = None
    for i in range(k):
        # 创建一个与 max_logits 形状相同的张量，用于存储目标 logits
        target_logit = output_logits[0, -1, max_indices[i]]
        target_logit.backward(target_logit)
        # target_logit.backward(target_logit,retain_graph=True)

        # 反向传播计算梯度
        # max_logits.backward(max_logits)
        relevance = input_embeds.grad.float().sum(-1).cpu()[0].tolist()
        if relevance_current is None:
            relevance_current = relevance
        else:
            relevance_current = [relevance[i] + relevance_current[i] for i in range(len(relevance))]
        output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits

    relevance_scores.append([x / k for x in relevance_current])

    # 获取下一个 token
    next_token_id = max_indices[0].item()
    next_token = tokenizer.decode(next_token_id)
    next_token_id = torch.tensor([[next_token_id]])
    next_token_id = next_token_id.to(model.device)
    print(next_token)
    generated_tokens.append(next_token)

    # 检查是否达到终止符
    if next_token == tokenizer.eos_token or next_token == "":
        break

    input_ids = torch.cat((input_ids, next_token_id), dim=-1)
    input_ids = input_ids.to(model.device)
    input_embeds = model.get_input_embeddings()(input_ids)

# 打印生成的序列
print(f"Generated sequence: {''.join(generated_tokens)}")

# 收集所有 token 和相关性分数
all_tokens = tokens + generated_tokens
all_relevance_align = []
for i in range(len(relevance_scores)):
    all_relevance_align.append(relevance_scores[i] + [0] * (len(relevance_scores[-1]) - len(relevance_scores[i])))
average_relevance = []
for i in range(len(all_relevance_align[0])):
    average_relevance.append(
        sum([x[i] for x in all_relevance_align]) / len([x[i] for x in all_relevance_align if x != 0]))

# average_relevance = torch.cat((relevance_scores[-1],torch.tensor([0])),dim=-1)
print(len(all_tokens))
print(len(average_relevance))
print(average_relevance)

# 清理 token 字符串
all_tokens = clean_tokens(all_tokens)

average_relevance = torch.tensor(average_relevance)

# 归一化相关性分数
average_relevance = average_relevance / average_relevance.abs().max()

# 生成热图
pdf_heatmap(all_tokens[:-1], average_relevance, path='heatmap.pdf', backend='xelatex')
