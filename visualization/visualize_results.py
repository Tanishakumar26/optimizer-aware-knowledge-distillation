import matplotlib.pyplot as plt


def plot_optimizer_accuracy():

    optimizers = ["AdamW", "Adam", "SGD", "RMSprop"]
    accuracies = [0.81, 0.86, 0.86, 0.52]

    plt.figure(figsize=(8, 5))

    plt.bar(optimizers, accuracies)

    plt.title("Optimizer Comparison - Validation Accuracy")
    plt.xlabel("Optimizers")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    for i, value in enumerate(accuracies):
        plt.text(i, value + 0.02, f"{value:.2f}", ha='center')

    plt.show()
def plot_loss_curves():

    epochs = [1, 2, 3]

    adamw_loss = [0.1946, 0.1720, 0.1671]
    adam_loss = [0.1950, 0.1723, 0.1689]
    sgd_loss = [0.1953, 0.1808, 0.1739]
    rmsprop_loss = [0.2691, 0.2137, 0.2113]

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, adamw_loss, marker='o', label="AdamW")
    plt.plot(epochs, adam_loss, marker='o', label="Adam")
    plt.plot(epochs, sgd_loss, marker='o', label="SGD")
    plt.plot(epochs, rmsprop_loss, marker='o', label="RMSprop")

    plt.title("Training Loss Curve Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks([1, 2, 3])
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    plot_optimizer_accuracy()
    plot_loss_curves()
    