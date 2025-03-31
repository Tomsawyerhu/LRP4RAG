import jsonlines

# 输入和输出文件路径
input_file = 'databricks-dolly-15k.jsonl'
output_file = 'databricks-dolly-15k_filtered.jsonl'

# 读取输入文件并过滤 context 字段为空的记录
with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
    for record in reader:
        if record.get('context'):
            writer.write(record)

print(f"Filtered records have been written to {output_file}")
