from datasets import load_dataset
import json
from openai import OpenAI
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

client = OpenAI()

model = "meta-llama/Llama-2-7b-chat-hf"

role_prompt = """
You are an assistant that only responds in json. You solve riddles and brain teasers that require complex reasoning.
Solve the riddle/brain teaser by selecting the correct option from the given option_list.
The response json should be in the format {"option_index": array index of the option selected from option_list. this should be a zero-based index , "option_answer": The answer selected from the given option_list} 
I only want the json output of this.
"""

test_data_files = {
    "SP": "brainteaser_test_data/SP_new_test.json",
    "WP": "brainteaser_test_data/WP_new_test.json"
}
test_dataset = load_dataset("json", data_files=test_data_files)

def prompt_generator(example):
    question = example["question"]
    option_list = example["choice_list"]
    prompt = f"""
Solve this brain teaser: {question}
option_list: {option_list}
[/INST]
"""
    example["prompt"] = prompt
    return example

test_dataset = test_dataset.map(prompt_generator)

def complete_chat(prompt_inst, prompt_user, model_id):
    response = client.chat.completions.create(
        model=model_id,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": prompt_inst},  # Instructions with the role as 'system'
            {"role": "user", "content": prompt_user},  # User's question with the role as 'user'
        ]
    )
    return json.loads(response.model_dump_json())

sp_list = []
for prompt in tqdm(KeyDataset(test_dataset["SP"], "prompt")):
    model = "gpt-4-1106-preview"
    response = complete_chat(
        prompt_inst=role_prompt,
        prompt_user=prompt,
        model_id=model
    )
    msg = response.get("choices")[0].get("message")
    content = msg.get("content")
    content = json.loads(content)
    option_index = content.get("option_index")
    sp_list.append(option_index)

wp_list = []
for prompt in tqdm(KeyDataset(test_dataset["WP"], "prompt")):
    model = "gpt-4-1106-preview"
    response = complete_chat(
        prompt_inst=role_prompt,
        prompt_user=prompt,
        model_id=model
    )
    msg = response.get("choices")[0].get("message")
    content = msg.get("content")
    content = json.loads(content)
    option_index = content.get("option_index")
    wp_list.append(option_index)

print(sp_list)
print(wp_list)

def file_w(path, array):
    with open(path, 'w') as file:
        for item in array:
            file.write(f"{item}" + '\n')
    print(path)

file_w("ans_gpt/answer_sen.txt", sp_list)
file_w("ans_gpt/answer_word.txt", wp_list)
