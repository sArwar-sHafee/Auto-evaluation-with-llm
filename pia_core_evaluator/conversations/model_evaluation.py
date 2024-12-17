import os
import json
from tqdm import tqdm
from typing import Dict, List
from .model_metrics import calculate_conversation_similarity
from .utils import extract_messages_from_conversation


def evaluate_model_metrics(
    ground_truth_dir: str, predictions_dir: str
) -> Dict[str, float]:
    """
    Evaluate model metrics for conversations in the given directories.

    Args:
        ground_truth_dir (str): Directory containing ground truth conversation JSON files.
        predictions_dir (str): Directory containing predicted conversation JSON files.

    Returns:
        Dict[str, float]: A dictionary containing average model metric scores and individual conversation scores.
    """
    total_similarity = 0.0
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

            similarity = calculate_conversation_similarity(gt_messages, pred_messages)
            total_similarity += similarity

            individual_scores.append({"filename": filename, "similarity": similarity})

            conversation_count += 1

    # Calculate average similarity
    avg_similarity = (
        total_similarity / conversation_count if conversation_count > 0 else 0
    )

    return {
        "average_similarity": avg_similarity,
        "individual_scores": individual_scores,
        "total_conversations": conversation_count,
    }
