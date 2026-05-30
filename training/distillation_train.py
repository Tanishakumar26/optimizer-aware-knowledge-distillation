import torch
from torch.optim import AdamW, Adam, SGD, RMSprop

from student.student_model import load_student_model
from teacher.teacher_model import load_teacher_model
from data.data_loader import load_and_tokenize_data
from training.distillation_loss import distillation_loss


def train_student_model(
    optimizer_name="adamw",
    epochs=8,
    learning_rate=2e-5
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load models
    student = load_student_model().to(device)
    teacher = load_teacher_model().to(device)

    # modes
    student.train()
    teacher.eval()

    # optimizer selection
    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(student.parameters(), lr=learning_rate)

    elif optimizer_name.lower() == "adam":
        optimizer = Adam(student.parameters(), lr=learning_rate)

    elif optimizer_name.lower() == "sgd":
        optimizer = SGD(student.parameters(), lr=0.01)

    elif optimizer_name.lower() == "rmsprop":
        optimizer = RMSprop(student.parameters(), lr=0.001)

    else:
        raise ValueError("Unsupported optimizer")

    # load dataset
    data = load_and_tokenize_data()

    print(f"Training started using {optimizer_name.upper()}...\n")

    for epoch in range(epochs):

        total_loss = 0

        # use subset first
        for i in range(5000):

            sample = data["train"][i]

            input_ids = torch.tensor([sample["input_ids"]]).to(device)
            attention_mask = torch.tensor([sample["attention_mask"]]).to(device)
            labels = torch.tensor([sample["label"]]).to(device)

            # teacher forward pass
            with torch.no_grad():
                teacher_output = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                teacher_probs = torch.softmax(
                    teacher_output.logits,
                    dim=-1
                )

            # student forward pass
            student_output = student(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # compute loss
            loss = distillation_loss(
                student_output.logits,
                teacher_probs,
                labels
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / 5000

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    print("\nTraining completed successfully.")

    student.save_pretrained("results/trained_student")
    print("Model saved successfully.")

    return student

if __name__ == "__main__":
    train_student_model()