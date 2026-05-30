import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_probs, true_labels, alpha=0.7):

    # KL divergence loss
    distill_loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        teacher_probs,
        reduction="batchmean"
    )

    # cross entropy loss
    ce_loss = F.cross_entropy(student_logits, true_labels)

    # final combined loss
    total_loss = alpha * distill_loss + (1 - alpha) * ce_loss

    return total_loss


if __name__ == "__main__":

    student_logits = torch.tensor([[0.2, 0.8]])
    teacher_probs = torch.tensor([[0.4571, 0.5429]])
    true_labels = torch.tensor([1])

    loss = distillation_loss(
        student_logits,
        teacher_probs,
        true_labels
    )

    print("Distillation loss:", loss.item())
    