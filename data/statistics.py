def main1():
    import jsonlines

    # 初始化计数器
    bins = 10
    bin_size = 200
    normal_counts = [0] * bins
    hallucination_counts = [0] * bins

    with jsonlines.open("./databricks-dolly-15k_judged_2.jsonl") as reader:
        for record in reader:
            context_length = len(record["context"])
            bin_index = min(context_length // bin_size, bins - 1)

            if record["label"] != "CORRECT":
                hallucination_counts[bin_index] += 1
            else:
                normal_counts[bin_index] += 1

    # 打印结果
    for i in range(bins):
        start = i * bin_size
        end = (i + 1) * bin_size - 1 if i < bins - 1 else "2000+"
        print(f"Range {start}-{end}: Normal: {normal_counts[i]}, Hallucinations: {hallucination_counts[i]}, Normal Ratio: {normal_counts[i]/(normal_counts[i]+hallucination_counts[i])}")

def main2():
    import matplotlib.pyplot as plt

    # 数据
    ranges = [
        "0-199", "200-399", "400-599", "600-799", "800-999",
        "1000-1199", "1200-1399", "1400-1599", "1600-1799", "1800-2000"
    ]
    normal_counts = [51, 194, 261, 246, 176, 125, 72, 60, 34, 46]
    hallucination_counts = [27, 85, 126, 141, 104, 85, 53, 49, 27, 38]
    print(sum(normal_counts),sum(hallucination_counts))
    normal_ratios = [
        normal_counts[i]/(normal_counts[i]+hallucination_counts[i]) for i in range(len(normal_counts))
    ]
    hallucination_ratio = [1 - x for x in normal_ratios]

    # 设置图形大小
    plt.figure(figsize=(14, 8))

    # 绘制柱状图
    bar_width = 0.7
    index = range(len(ranges))

    bars_normal = plt.bar(index, normal_counts, bar_width, label='Normal', color='lightblue')
    bars_hallucination = plt.bar(index, hallucination_counts, bar_width, bottom=normal_counts, label='Hallucination', color='orange')

    # 添加样本数量的文本
    for i, (normal_count, hallucination_count) in enumerate(zip(normal_counts, hallucination_counts)):
        total_height = normal_count + hallucination_count
        plt.text(i, normal_count / 2, f'{normal_count}', ha='center', va='center', color='black', fontsize=10)
        plt.text(i, normal_count + hallucination_count / 2, f'{hallucination_count}', ha='center', va='center', color='black', fontsize=10)

    # 添加幻觉比率的文本
    for i, ratio in enumerate(hallucination_ratio):
        plt.text(i, normal_counts[i]+hallucination_counts[i] + 10, f'{ratio:.2f}', ha='center', va='bottom', fontsize=10)

    # 设置x轴标签
    plt.xticks(index, ranges)
    plt.xlabel('Context Length')
    plt.ylabel('Number of Samples')
    # plt.title('Normal vs. Hallucinations by Context Length Range')
    plt.legend()

    # 导出为PDF文件
    plt.tight_layout()
    plt.savefig("./distribution.pdf")

    # 显示图形
    plt.show()

def main3():
    import jsonlines
    import random
    # 读取原始数据
    with jsonlines.open("./databricks-dolly-15k_judged.jsonl") as reader:
        data = list(reader)

    # 筛选出符合条件的记录
    filtered_data = [record for record in data if record["label"] == "CORRECT" and record["judge"] == "no"]

    # 随机选择 400 个记录进行移除
    if len(filtered_data) >= 1170:
        to_remove = random.sample(filtered_data, 1171)
    else:
        print("Warning: Less than 400 records meet the criteria. Removing all matching records.")
        to_remove = filtered_data

    # 移除选中的记录
    for record in to_remove:
        data.remove(record)

    # 筛选出符合条件的记录
    filtered_data = [record for record in data if record["label"] != "CORRECT" and record["judge"] == "no"]
        # 随机选择 400 个记录进行移除
    if len(filtered_data) >= 500:
        to_remove = random.sample(filtered_data, 500)
    else:
        print("Warning: Less than 400 records meet the criteria. Removing all matching records.")
        to_remove = filtered_data

        # 移除选中的记录
    for record in to_remove:
        data.remove(record)

    # 写入新的文件
    with jsonlines.open("./databricks-dolly-15k_judged_2.jsonl", "w") as writer:
        writer.write_all(data)
def calculate_metrics(TN, TP, FN, FP):
    # 计算准确率
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # 计算精确率
    precision = TP / (TP + FP)
    # 计算召回率
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        "True Positive": TP,
        "True Negative": TN,
        "False Positive": FP,
        "False Negative": FN,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }

def main4():
    import jsonlines
    FP,TP,FN,TN=0,0,0,0
    with jsonlines.open("./databricks-dolly-15k_judged_ablation_2.jsonl") as reader:
        for record in reader:

            if "yes" == record["judge"]:
                if record["label"] == "CORRECT":
                    print("no hallucination, judge wrong")
                    FP += 1
                else:
                    print("exists hallucination, judge correct")
                    TN += 1
            else:
                if record["label"] == "CORRECT":
                    print("no hallucination, judge correct")
                    TP += 1
                else:
                    print("exists hallucination, judge wrong")
                    FN += 1
    print(calculate_metrics(TN, TP, FN, FP))

def main5():
    import jsonlines
    # 读取原始数据
    with jsonlines.open("./databricks-dolly-15k_judged_2.jsonl") as reader,jsonlines.open("./databricks-dolly"
                                                                                          "-15k_judged_3.jsonl",
                                                                                          "w") as writer:
        for record in reader:
            writer.write({
                "context":record["context"],
                "instruction":record["instruction"],
                "response":record["response"],
                "label": record["label"],
                "judge": record["judge"]
            })

def main6():
    import jsonlines
    with jsonlines.open("./databricks-dolly-15k_judged_2.jsonl") as reader,jsonlines.open("./databricks-dolly-15k_judged_4.jsonl","a") as writer:
        for record in reader:
            new_record={
                "context": record["context"],
                "instruction": record["instruction"],
                "answer": record["answer"],
                "response": record["response"],
                "label": record["label"],
            }
            writer.write(new_record)

def main7():
    import jsonlines
    ablation_result,orig_result={},{}
    with jsonlines.open("./databricks-dolly-15k_judged_ablation.jsonl") as reader:
        for record in reader:
            ablation_result[record["context"]]=(record["instruction"],record["label"],record["answer"],record["judge"],record["context_short_lrp"],record["context_short_llm"],record["response"])

    with jsonlines.open("./databricks-dolly-15k_judged_2.jsonl") as reader:
        for record in reader:
            orig_result[record["context"]]=(record["instruction"],record["label"],record["answer"],record["judge"],record["context_short_lrp"],record["context_short_llm"],record["response"])
    for c in ablation_result.keys():
        ablation_record=ablation_result[c]
        if ablation_result[c][1]!="CORRECT" and  ablation_result[c][3]=="no":
            orig_record=orig_result[c]
            if orig_record[3]!="no":
                print(1)

def main8():
    import jsonlines
    answer_length=[]
    llm_context_length=[]
    lrp_context_length=[]

    with jsonlines.open("./databricks-dolly-15k_judged_2.jsonl") as reader:
        for record in reader:
            answer_length.append(len(record["relevance_scores"][-1])-len(record["relevance_scores"][0]))
    print(sum(answer_length)/len(answer_length))

main8()
