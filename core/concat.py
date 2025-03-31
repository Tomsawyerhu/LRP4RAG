import jsonlines

input_files=[
    "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_output.jsonl",
    "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_output2.jsonl",
    "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_output3.jsonl",
    "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_output4.jsonl"]

output_file = "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_concat.jsonl"

contexts=set()
with jsonlines.open(output_file,'w') as writer:
    for input_file in input_files:
        with jsonlines.open(input_file) as reader:
            for record in reader:
                if len(record["context"])>2000:
                    continue
                elif record["context"] in contexts:
                    continue
                else:
                    writer.write(record)