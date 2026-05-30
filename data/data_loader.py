from datasets import load_dataset
from transformers import AutoTokenizer


def load_and_tokenize_data():

    # load SST-2 dataset
    dataset = load_dataset("glue", "sst2")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # tokenization function
    def tokenize_function(example):
        return tokenizer(
            example["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    # apply tokenizer
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True
    )

    return tokenized_dataset


if __name__ == "__main__":

    data = load_and_tokenize_data()

    print(data)
    print(data["train"][0])