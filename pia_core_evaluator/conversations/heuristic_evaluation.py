import os
import json
from loguru import logger
from tqdm import tqdm
from typing import Dict, Callable
from .heuristic_mertics import Heuristic
from .utils import extract_messages_from_conversation


def evaluate_heuristics(
    ground_truth_dir: str, predictions_dir: str, tokenizer: Callable = None
) -> Dict[str, float]:
    """
    Evaluate heuristics for conversations in the given directories.

    Args:
        ground_truth_dir (str): Directory containing ground truth conversation JSON files.
        predictions_dir (str): Directory containing predicted conversation JSON files.

    Returns:
        Dict[str, float]: A dictionary containing average heuristic scores and individual conversation scores.
    """
    heuristic = Heuristic(tokenizer=tokenizer)
    total_scores = {"bleu": 0, "sacrebleu": 0, "rouge1": 0, "rougeL": 0, "meteor": 0}
    individual_scores = []
    conversation_count = 0

    for filename in tqdm(os.listdir(ground_truth_dir)):
        if filename.endswith(".json"):
            gt_path = os.path.join(ground_truth_dir, filename)
            pred_path = os.path.join(predictions_dir, filename)

            if not os.path.exists(pred_path):
                continue

            with open(gt_path, "r") as gt_file, open(pred_path, "r") as pred_file:
                gt_conversation = json.load(gt_file)
                pred_conversation = json.load(pred_file)

            gt_messages = extract_messages_from_conversation(
                gt_conversation, role="assistant"
            )
            pred_messages = extract_messages_from_conversation(
                pred_conversation, role="assistant"
            )
            logger.info(f"gt_messages_len: {len(gt_messages)}")
            logger.info(f"pred_messages_len: {len(pred_messages)}")

            scores = heuristic(gt_messages, pred_messages)

            for metric, score in scores.items():
                total_scores[metric] += score

            individual_scores.append({"filename": filename, "scores": scores})

            conversation_count += 1

    # Calculate average scores
    avg_scores = {
        metric: total / conversation_count if conversation_count > 0 else 0
        for metric, total in total_scores.items()
    }

    return {
        "average_scores": avg_scores,
        "individual_scores": individual_scores,
        "total_conversations": conversation_count,
    }
