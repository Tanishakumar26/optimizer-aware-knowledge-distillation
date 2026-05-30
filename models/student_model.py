from transformers import AutoModelForSequenceClassification
import torch


def load_student_model():

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model


if __name__ == "__main__":

    student = load_student_model()

    total_params = sum(p.numel() for p in student.parameters())

    print("Student model loaded successfully")
    print("Total parameters:", total_params)