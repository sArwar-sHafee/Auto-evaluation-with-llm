import os
import json
from typing import Dict
from tqdm import tqdm
from loguru import logger
from .utils import (
    extract_functions_from_conversation,
    validate_function_flow,
    validate_function_accuracy,
)


def evaluate_tool_calling(
    ground_truth_dir: str, predictions_dir: str, match_arg_value: bool = True
) -> Dict[str, float]:
    """
    Evaluate tool calling accuracy by comparing ground truth and predicted conversations.

    Args:
        ground_truth_dir (str): Directory containing ground truth conversation JSON files.
        predictions_dir (str): Directory containing predicted conversation JSON files.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics.
    """
    total_conversations = 0
    correct_flows = 0
    all_validation_results = []
    total_accuracy = 0.0

    for filename in tqdm(os.listdir(ground_truth_dir)):
        if filename.endswith(".json"):
            gt_path = os.path.join(ground_truth_dir, filename)
            pred_path = os.path.join(predictions_dir, filename)

            if not os.path.exists(pred_path):
                continue

            with open(gt_path, "r") as gt_file, open(pred_path, "r") as pred_file:
                gt_conversation = json.load(gt_file)
                pred_conversation = json.load(pred_file)

            gt_functions = extract_functions_from_conversation(gt_conversation)
            pred_functions = extract_functions_from_conversation(pred_conversation)
            logger.info(f"gt_functions: {gt_functions}")
            logger.info(f"pred_functions: {pred_functions}")

            flow_validation_result = validate_function_flow(
                gt_functions, pred_functions, match_arg_value=match_arg_value
            )
            function_validation_result = validate_function_accuracy(
                gt_functions, pred_functions, match_arg_value=match_arg_value
            )
            logger.info(f"flow_validation_result: {flow_validation_result}")
            logger.info(f"accuracy_validation_result: {function_validation_result}")

            # Combine flow and accuracy validation results
            validation_result = {
                "filename": filename,
                "flow_validation": flow_validation_result,
                "accuracy_validation": function_validation_result,
            }
            all_validation_results.append(validation_result)

            if flow_validation_result["valid_flow"]:
                correct_flows += 1
            total_conversations += 1
            total_accuracy += function_validation_result["accuracy"]

    flow_accuracy = (
        correct_flows / total_conversations if total_conversations > 0 else 0
    )
    average_function_validation_accuracy = (
        total_accuracy / total_conversations if total_conversations > 0 else 0
    )

    return {
        "flow_accuracy": flow_accuracy,
        "average_function_validation_accuracy": average_function_validation_accuracy,
        "total_conversations": total_conversations,
        "correct_flows": correct_flows,
        "individual_scores": all_validation_results,
    }
