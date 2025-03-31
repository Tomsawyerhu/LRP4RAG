import openai

openai.api_base = "https://api.nextapi.fun/v1"
openai.api_key = "ak-xHR4emmF67DYoeAdupb1y75ZVNxMax4x9uxdWC9E4eWxvxLB"


# def make_model(base_url="https://api.nextapi.fun/v1", api_key="ak-xHR4emmF67DYoeAdupb1y75ZVNxMax4x9uxdWC9E4eWxvxLB"):
#     _client = OpenAI(
#         base_url=base_url,
#         api_key=api_key,
#     )
#     return _client
#
#
# gpt_client = make_model()


def generate(prompt, model_name="gpt-4o-mini", timeout=100, retry=10):
    for i in range(retry):
        completion = openai.ChatCompletion.create(
            model=model_name, messages=[{"role": "user", "content": prompt}], timeout=timeout
        )

        if not completion.choices or len(completion.choices) == 0:
            continue
        else:
            text = completion.choices[0].message.content
            return text
    print("No reply from GPT")
    return ""