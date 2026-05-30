import torch
import torch.nn.functional as F

from teacher.teacher_model import load_teacher_model
from data.data_loader import load_and_tokenize_data


def generate_teacher_outputs():

    teacher = load_teacher_model()

    data = load_and_tokenize_data()

    sample = data["train"][0]

    input_ids = torch.tensor([sample["input_ids"]])
    attention_mask = torch.tensor([sample["attention_mask"]])

    with torch.no_grad():
        outputs = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    probabilities = F.softmax(outputs.logits, dim=-1)
    return probabilities


if __name__ == "__main__":

    probs = generate_teacher_outputs()

    print("Teacher soft probabilities:")
    print(probs)
