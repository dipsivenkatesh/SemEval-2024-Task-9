from datasets import load_dataset
test_data_files = {
    "SP": "brainteaser_test_data/SP_new_test.json",
    "WP": "brainteaser_test_data/WP_new_test.json"
}
test_dataset = load_dataset("json", data_files=test_data_files)
print(test_dataset)
