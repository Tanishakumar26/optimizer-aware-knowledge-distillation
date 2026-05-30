from training.distillation_train import train_student_model
from evaluation.evaluate import evaluate_student_model


def run_optimizer_comparison():

    results = {}

    for opt in ["adamw", "adam" , "sgd", "rmsprop"]:

        print(f"\n{'='*50}")
        print(f"Running experiment with {opt.upper()}")
        print(f"{'='*50}\n")

        train_student_model(optimizer_name=opt)

        accuracy = evaluate_student_model()

        results[opt] = accuracy

    print("\nFinal Comparison Results:")
    print(results)

    return results


if __name__ == "__main__":
    run_optimizer_comparison()
    