import numpy as np
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset


def npy_to_json(filename):
    data = np.load(filename, allow_pickle=True)
    data = data.tolist()
    json_str = json.dumps(data)

    base_name = os.path.splitext(filename)[0]
    json_filename = base_name + '.json'
    print(json_filename)

    with open(json_filename, 'w') as json_file:
        json_file.write(json_str)

def npy_to_list(filename):
    data = np.load(filename, allow_pickle=True)
    data = data.tolist()
    return data

files = [
    "BrainTeaser_data/data/SP_eval_data_for_practice.npy",
    "BrainTeaser_data/data/SP-train.npy",
    "BrainTeaser_data/data/WP_eval_data_for_practice.npy",
    "BrainTeaser_data/data/WP-train.npy"
]

files = [
    "brainteaser_test_data/SP_new_test.json",
    "brainteaser_test_data/WP_new_test.json"
]
test_data_files = {
    "SP": "brainteaser_test_data/SP_new_test.json",
    "WP": "brainteaser_test_data/WP_new_test.json"
}
test_dataset = load_dataset("json", data_files=test_data_files)
print(test_dataset)
def prompt_generator(example):
    question = example["question"]
    option_list = example["choice_list"]
    prompt_inst = """
<s>[INST] <<SYS>>
You are an assistant that only respods in json. You solve riddles and brain teasers that require complex reasoning.
Solve the riddle by selecting the correct option from the given option_list
The response json shold be in the format {"option_index": array index of the option selected, this should be a zero-based index, "option_answer": The answer selected from the given option_list} 
<</SYS>>
"""
    prompt_user = f"""
Solve this brain teaser: {question}
option_list: {option_list}
[/INST]
"""
    prompt = prompt_inst + "\n" + prompt_user
    example["prompt"] = prompt
    return example

test_dataset = test_dataset.map(prompt_generator)
print(test_dataset["SP"][51]["question"])
print(test_dataset["SP"][51]["choice_list"])
# print((KeyDataset(test_dataset["SP"], "prompt")))
# for prompt_one in (KeyDataset(test_dataset["SP"], "prompt")):
#     print(prompt_one)
#     break


# for file in files:
#     # npy_to_json(file)
#     dataset = load_dataset("json", data_files=file)
#     print(dataset)


# train_list = npy_to_list("BrainTeaser_data/data/SP-train.npy")
# print(len(train_list))
# pbar = tqdm(total=len(train_list), desc="Processing items")
# for item in train_list:
#     pbar.update(1)
# # Your for loop
