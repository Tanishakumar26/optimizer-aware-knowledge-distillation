import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from transformers import AutoModelForSequenceClassification
from data.data_loader import load_and_tokenize_data


def evaluate_student_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load trained student
    student = AutoModelForSequenceClassification.from_pretrained(
    "results/trained_student"
    ).to(device)
    student.eval()

    # load data
    data = load_and_tokenize_data()

    true_labels = []
    predicted_labels = []

    print("Evaluation started...\n")

    # evaluate on validation set
    for i in range(100):
        sample = data["validation"][i]

        input_ids = torch.tensor([sample["input_ids"]]).to(device)
        attention_mask = torch.tensor([sample["attention_mask"]]).to(device)
        label = sample["label"]

        with torch.no_grad():
            output = student(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            prediction = torch.argmax(output.logits, dim=-1).item()

        true_labels.append(label)
        predicted_labels.append(prediction)

    # accuracy
    acc = accuracy_score(true_labels, predicted_labels)

    print(f"Validation Accuracy: {acc:.4f}")

    # confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print("\nConfusion Matrix:")
    print(cm)

    # classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))

    return acc


if __name__ == "__main__":
    evaluate_student_model()
