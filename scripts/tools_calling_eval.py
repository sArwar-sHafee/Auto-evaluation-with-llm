import sys

sys.path.append("../")
from pprint import pprint
from pia_core_evaluator.tools_calling.tool_calling_evaluation import (
    evaluate_tool_calling,
)


def main():
    ground_truth_dir = "../data/sample_data/gt"
    predictions_dir = "../data/sample_data/pred"

    evaluation_results = evaluate_tool_calling(ground_truth_dir, predictions_dir)
    pprint(evaluation_results)


if __name__ == "__main__":
    main()