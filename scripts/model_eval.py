import sys

sys.path.append("../")
from pprint import pprint
from pia_core_evaluator.conversations.model_evaluation import evaluate_model_metrics


def main():
    ground_truth_dir = "../data/sample_data/gt"
    predictions_dir = "../data/sample_data/pred"

    evaluation_results = evaluate_model_metrics(ground_truth_dir, predictions_dir)
    pprint(evaluation_results)


if __name__ == "__main__":
    main()
