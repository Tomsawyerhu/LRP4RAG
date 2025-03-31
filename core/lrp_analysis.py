import jsonlines
from transformers import AutoTokenizer
import torch

from utils import generate

path = "/mnt/data/hhc/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(path)

find_relevant_context_prompt=lambda context,question:f"""
Context: {context}
Question: {question}
Please extract the sentences from the context that are most relevant and helpful in answering the question. 
Omit sentences that are not important and keep it as concise as possible. 
You should only return a paragraph composed of the original sentences from the context, 
without summarizing or rewriting them.
"""

"""
根据lrp的相关性找出最有关联的context
"""
def extract_most_relevant_context(relevance_scores, source_text, top_k_percent=50, question=None):
    # 收集关于输入token的相关性分数
    all_relevance_align = []
    for i in range(len(relevance_scores)):
        all_relevance_align.append(relevance_scores[i][:len(relevance_scores[0])])
    average_relevance = []
    for i in range(len(all_relevance_align[0])):
        average_relevance.append(
            sum([x[i] for x in all_relevance_align]) / len([x[i] for x in all_relevance_align]))
    # print(len(average_relevance))

    source_token_ids = tokenizer(source_text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    source_tokens = tokenizer.convert_ids_to_tokens(source_token_ids)
    assert len(source_tokens) == len(average_relevance)

    average_relevance = torch.tensor(average_relevance)
    # 归一化相关性分数
    average_relevance = average_relevance / average_relevance.abs().max()
    average_relevance = average_relevance.tolist()

    # # 计算要提取的top k个token的数量
    # top_k = max(1, int(len(average_relevance) * (top_k_percent / 100.0)))
    #
    # # 获取相关性最高的top k个token的索引
    # top_k_indices = sorted(range(len(average_relevance)), key=lambda i: average_relevance[i], reverse=True)[:top_k]

    ## 计算每个句子的重要性
    all_sentence_tokens = []
    current_sentence_tokens = []
    current_relevance = []
    for i, token in enumerate(source_tokens):
        current_sentence_tokens.append(token)
        current_relevance.append(average_relevance[i])
        if token == '.' or token == ',' or i == len(source_tokens) - 1:
            all_sentence_tokens.append((current_sentence_tokens, current_relevance))
            current_sentence_tokens = []
            current_relevance = []

    # 计算每个句子的平均相关性分数
    sentence_importance = []
    sentences = []

    for sentence_tokens, relevance in all_sentence_tokens:
        sentence = tokenizer.convert_tokens_to_string(sentence_tokens)
        sentences.append(sentence)
        sentence_len = len(sentence.split())
        sentence_importance.append(sum(relevance) / sentence_len)

    # 选择相关性最高的前top k%的句子
    top_k = max(1, int(len(sentence_importance) * (top_k_percent / 100.0)))
    top_k_indices = sorted(range(len(sentence_importance)), key=lambda i: sentence_importance[i], reverse=True)
    indices_selected = top_k_indices[:top_k]

    # 提取这些句子
    relevant_sentences = []
    for i, sentence in enumerate(sentences):
        if i in indices_selected:
            relevant_sentences.append(sentence)

    short_context = ''.join(relevant_sentences)
    short_context = short_context.replace("Read the above context and answer the following question.", "")
    if question is not None:
        short_context = short_context.replace(question, "")
    return short_context

def llm_judge_most_relevant_context(source_context,question):
    return generate(find_relevant_context_prompt(source_context,question))

def main1():
    input_file = "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_output.jsonl"
    output_file = "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_short_context_lrp.jsonl"
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, 'w') as writer:
        for record in reader:
            relevant_scores = record["relevance_scores"]
            source_text = record["prompt"]
            print("----------source--------------")
            print(source_text)
            print("----------short--------------")
            print(extract_most_relevant_context(relevant_scores, source_text, 50, record["instruction"]))
            print("----------question--------------")
            print(record["instruction"])
            print("----------gt--------------")
            print(record["response"])
            print("----------answer--------------")
            print(record["answer"])
            record["context_short_lrp"]=extract_most_relevant_context(relevant_scores, source_text, 50, record["instruction"])
            writer.write(record)
            # break

def main2():
    input_file = "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_short_context_lrp.jsonl"
    output_file = "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_short_context_llm.jsonl"
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, 'w') as writer:
        for record in reader:
            question = record["instruction"]
            context = record["context"]
            print("----------source--------------")
            print(context)
            print("----------short--------------")
            context_short=llm_judge_most_relevant_context(context, question)
            print(context_short)
            print("----------question--------------")
            print(record["instruction"])
            print("----------gt--------------")
            print(record["response"])
            print("----------answer--------------")
            print(record["answer"])
            record["context_short_llm"] = context_short
            writer.write(record)
            # break

def main3():
    reference_file = "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_short_context_llm.jsonl"
    input_file = "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_short_context_lrp_k20.jsonl"
    output_file = "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_short_context_llm_k20.jsonl"
    llm_contexts={}
    with jsonlines.open(reference_file) as reference_reader:
        for record in reference_reader:
            llm_contexts[(record["context"],record["instruction"])]=record["context_short_llm"]
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, 'w') as writer:
        for record in reader:
            record["context_short_llm"]=llm_contexts[(record["context"],record["instruction"])]
            writer.write(record)

if __name__ == '__main__':
    main2()


# Acc: 0.2165
# AUROC-Perplexity: 0.6002807130757442
# AUROC-Energy: 0.5969014008765561
# AUROC-Entropy: 0.5523798768519266
# AUROC-LexicalSim: 0.6176047968594554
# AUROC-SentBertScore: 0.5
# AUROC-EigenScore: 0.5045220617891425
# AUROC-EigenScore-Output: 0.5971761985426582
# rho_Perplexity: 0.3087409270733522
# rho_Energy: -0.08480735447320946
# rho_Entropy: 0.23476080752670386
# rho_LexicalSimilarity: 0.41038372841663234
# rho_EigenScore: 0.2112426911241919
# rho_EigenScoreOutput: 0.2901266990951345
# TruthfulQA Perplexity threshold: -0.2213
# TruthfulQA Energy threshold: 26.6272
# TruthfulQA Entropy threshold: -0.1362
# TruthfulQA LexicalSimilarity threshold: 0.3906
# TruthfulQA SentBertScore threshold: inf
# TruthfulQA EigenIndicator threshold: 1.6042
# TruthfulQA EigenIndicatorOutput threshold: 1.1346
# Perplexity Metrics: {'accuracy': 0.565, 'precision': 0.6908212560386473, 'recall': 0.5652173913043478, 'f1': 0.6217391304347826}
# Energy Metrics: {'accuracy': 0.5715, 'precision': 0.703187250996016, 'recall': 0.558102766798419, 'f1': 0.6223005729396209}
# Entropy Metrics: {'accuracy': 0.5535, 'precision': 0.6684782608695652, 'recall': 0.583399209486166, 'f1': 0.6230476994512452}
# LexicalSimilarity Metrics: {'accuracy': 0.609, 'precision': 0.703111858704794, 'recall': 0.6608695652173913, 'f1': 0.6813365933170334}
# EigenIndicator Metrics: {'accuracy': 0.515, 'precision': 0.6351970669110908, 'recall': 0.5478260869565217, 'f1': 0.5882852292020373}
# EigenIndicatorOutput Metrics: {'accuracy': 0.5915, 'precision': 0.6888701517706577, 'recall': 0.6458498023715415, 'f1': 0.6666666666666667}