import os.path
import tqdm
import jsonlines

from utils import generate

hallucination_types = ["IN_CONTEXT", "COUNTER_FACTUAL", "IRRELEVANT", "CORRECT"]

check_correct_prompt_template = lambda question, gt_answer, answer_to_check: f"""
The question is: {question}
The ground-truth answer is: {gt_answer}
A user has provided a different answer. Please determine if the answer is correct based on the question and the ground-truth answer.
The different answer provided by user is: {answer_to_check}
Is it correct? Do not explain, return YES or NO only.
"""

check_hallucination_type_template = lambda question, answer_to_check: f"""
There are three hallucination type:
COUNTER_FACTUAL. If the answer is inconsistent with factual information (for example, the Earth orbits the Sun, the Sun rises in the East).
IRRELEVANT. If the answer is irrelevant to the question.
IN_CONTEXT. If the answer is inconsistent with the context. Since we do not provide context, you can classify hallucinations that do not fall into the previous two categories as IN_CONTEXT.

The question is: {question}
The hallucinated answer is: {answer_to_check}
Return COUNTER_FACTUAL or IRRELEVANT or IN_CONTEXT according to the type of hallucination.
"""


def judge_if_hallucination_exists(question, gt_answer, answer_to_check):
    gpt_answer = generate(check_correct_prompt_template(question, gt_answer, answer_to_check))
    if "yes" in gpt_answer.lower():
        return False
    elif "no" in gpt_answer.lower():
        return True
    else:
        return None


def sanitize(answer):
    answer = answer.strip()
    if answer.startswith("?") or answer.startswith(",") or answer.startswith(".") or answer.startswith(
            ";") or answer.startswith(":"):
        answer = answer[1:]
    bullshits = ["You are an AI assistant", "You are a helpful assistant", "You are an assistant"]
    for bullshit in bullshits:
        if bullshit in answer:
            answer = answer.split(bullshit)[0]
    self_setting = "(If the question is unanswerable, say \"unanswerable\")"
    if self_setting in answer:
        answer = answer.split(self_setting)[1]
    answer = answer.strip()
    return answer


def judge_hallucination_type(question, answer_to_check):
    gpt_answer = generate(check_hallucination_type_template(question, answer_to_check))
    if "counter_factual" in gpt_answer.lower():
        return "COUNTER_FACTUAL"
    elif "irrelevant" in gpt_answer.lower():
        return "IRRELEVANT"
    elif "in_context" in gpt_answer.lower():
        return "IN_CONTEXT"
    else:
        return "UNKNOWN_TYPE"


if __name__ == '__main__':
    input_file = "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_concat.jsonl"
    output_file = "/mnt/data/hhc/lrp4rag/data/databricks-dolly-15k_labeled.jsonl"
    contexts = set()
    if os.path.exists(output_file):
        with jsonlines.open(output_file) as context_checker:
            for record in context_checker:
                contexts.add(record["context"])
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, 'w') as writer:
        for record in tqdm.tqdm(reader):
            if record["context"] in contexts:
                continue
            instruction = record["instruction"]
            gt_answer = record["response"]
            answer_to_check = record["answer"]
            answer_to_check = sanitize(answer_to_check)
            hallucination_exists = judge_if_hallucination_exists(instruction, gt_answer, answer_to_check)
            if hallucination_exists is None:
                continue
            elif hallucination_exists:
                hallucination_type = judge_hallucination_type(instruction, answer_to_check)
                record["label"] = hallucination_type
            else:
                record["label"] = "CORRECT"
            writer.write(record)
