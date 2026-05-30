from transformers import AutoModelForSequenceClassification
import torch

def load_teacher_model():

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    return model
if __name__ == "__main__":

    teacher = load_teacher_model()

    total_params = sum(p.numel() for p in teacher.parameters())

    print("Teacher model loaded successfully")
    print("Total parameters:", total_params)