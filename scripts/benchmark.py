import os
import json
import argparse
import sys

sys.path.append("../")
import pandas as pd

from pia_core_evaluator.tools_calling.tool_calling_evaluation import (
    evaluate_tool_calling,
)
from pia_core_evaluator.conversations.heuristic_evaluation import evaluate_heuristics
from pia_core_evaluator.conversations.model_evaluation import evaluate_model_metrics
from pia_core_evaluator.conversations.deepeval_evaluation import (
    evaluate_deepeval_metrics,
)

parser = argparse.ArgumentParser()
parser.add_argument("--tool_calling", action="store_true")
parser.add_argument("--heuristic", action="store_true")
parser.add_argument("--model", action="store_true")
parser.add_argument("--deepeval", action="store_true")
parser.add_argument("--match_function_arg_value", action="store_true")
parser.add_argument(
    "--ground_truth_file_path", type=str, default="../data/sample_data/gt"
)
parser.add_argument(
    "--predictions_file_path", type=str, default="../data/sample_data/pred"
)
parser.add_argument("--dump_path", type=str, default="../data")

args = parser.parse_args()


def benchmark():
    overall_results = {}

    os.makedirs(args.dump_path, exist_ok=True)

    if args.tool_calling:
        tool_eval_results = evaluate_tool_calling(
            ground_truth_dir=args.ground_truth_file_path,
            predictions_dir=args.predictions_file_path,
            match_arg_value=args.match_function_arg_value,
        )
        overall_results["tool_flow_accuracy"] = tool_eval_results["flow_accuracy"]
        overall_results[
            "tool_average_function_validation_accuracy"
        ] = tool_eval_results["average_function_validation_accuracy"]
        overall_results["tool_correct_flows"] = tool_eval_results["correct_flows"]
        overall_results["tool_total_conversations"] = tool_eval_results[
            "total_conversations"
        ]

        tool_result_dump_path = os.path.join(
            args.dump_path, "tool_calling_results.json"
        )
        with open(tool_result_dump_path, "w") as f:
            json.dump(tool_eval_results, f, ensure_ascii=False, indent=4)

    if args.heuristic:
        heuristic_eval_results = evaluate_heuristics(
            ground_truth_dir=args.ground_truth_file_path,
            predictions_dir=args.predictions_file_path,
        )
        overall_results["conversations_average_bleu_score"] = heuristic_eval_results[
            "average_scores"
        ]["bleu"]
        overall_results["conversations_average_meteor_score"] = heuristic_eval_results[
            "average_scores"
        ]["meteor"]
        overall_results["conversations_average_rouge1_score"] = heuristic_eval_results[
            "average_scores"
        ]["rouge1"]
        overall_results["conversations_average_rougeL_score"] = heuristic_eval_results[
            "average_scores"
        ]["rougeL"]
        overall_results[
            "conversations_average_sacrebleu_score"
        ] = heuristic_eval_results["average_scores"]["sacrebleu"]

        heuristic_result_dump_path = os.path.join(
            args.dump_path, "heuristic_results.json"
        )
        with open(heuristic_result_dump_path, "w") as f:
            json.dump(heuristic_eval_results, f, ensure_ascii=False, indent=4)

    if args.model:
        model_eval_results = evaluate_model_metrics(
            ground_truth_dir=args.ground_truth_file_path,
            predictions_dir=args.predictions_file_path,
        )
        overall_results["conversations_average_similarity"] = model_eval_results[
            "average_similarity"
        ]

        model_result_dump_path = os.path.join(args.dump_path, "model_results.json")
        with open(model_result_dump_path, "w") as f:
            json.dump(model_eval_results, f, ensure_ascii=False, indent=4)

    if args.deepeval:
        deepeval_eval_results = evaluate_deepeval_metrics(
            ground_truth_dir=args.ground_truth_file_path,
            predictions_dir=args.predictions_file_path,
        )
        overall_results[
            "conversations_average_answer_relevancy"
        ] = deepeval_eval_results["average_scores"]["answer_relevancy"]
        overall_results["conversations_average_correctness"] = deepeval_eval_results[
            "average_scores"
        ]["correctness"]

        deepeval_result_dump_path = os.path.join(
            args.dump_path, "deepeval_results.json"
        )
        with open(deepeval_result_dump_path, "w") as f:
            json.dump(deepeval_eval_results, f, ensure_ascii=False, indent=4)

    overall_results_df = pd.DataFrame([overall_results])
    dump_file_path = os.path.join(args.dump_path, "benchmark_results.csv")
    overall_results_df.to_csv(dump_file_path, index=False)


if __name__ == "__main__":
    benchmark()
