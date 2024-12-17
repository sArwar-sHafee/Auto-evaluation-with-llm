import os
import json
from tqdm import tqdm
from typing import Dict
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from tqdm import tqdm
from .deepeval_metrics import get_correctness_metric, get_answer_relevancy_metric
from .utils import extract_messages_from_conversation
import multiprocessing
from functools import partial


def process_conversation(filename, ground_truth_dir, predictions_dir, model):
    gt_path = os.path.join(ground_truth_dir, filename)
    pred_path = os.path.join(predictions_dir, filename)

    if not os.path.exists(pred_path):
        return None

    with open(gt_path, "r") as gt_file, open(pred_path, "r") as pred_file:
        gt_conversation = json.load(gt_file)
        pred_conversation = json.load(pred_file)

    gt_input_messages = extract_messages_from_conversation(gt_conversation, role="user")
    gt_output_messages = extract_messages_from_conversation(
        gt_conversation, role="assistant"
    )
    pred_output_messages = extract_messages_from_conversation(
        pred_conversation, role="assistant"
    )

    correctness_metric = get_correctness_metric(model=model)
    answer_relevancy_metric = get_answer_relevancy_metric(model=model)

    conversation_scores = {"correctness": [], "answer_relevancy": []}
    conversation_reasons = {"correctness": [], "answer_relevancy": []}

    for input, expected_output, actual_output in zip(
        gt_input_messages, gt_output_messages, pred_output_messages
    ):
        test_case = LLMTestCase(
            input=input, actual_output=actual_output, expected_output=expected_output
        )

        correctness_metric.measure(test_case)
        answer_relevancy_metric.measure(test_case)

        conversation_scores["correctness"].append(correctness_metric.score)
        conversation_scores["answer_relevancy"].append(answer_relevancy_metric.score)
        conversation_reasons["correctness"].append(correctness_metric.reason)
        conversation_reasons["answer_relevancy"].append(answer_relevancy_metric.reason)

    # Calculate average scores for this conversation
    avg_correctness_score = (
        sum(conversation_scores["correctness"])
        / len(conversation_scores["correctness"])
        if conversation_scores["correctness"]
        else 0
    )
    avg_relevancy_score = (
        sum(conversation_scores["answer_relevancy"])
        / len(conversation_scores["answer_relevancy"])
        if conversation_scores["answer_relevancy"]
        else 0
    )

    return {
        "filename": filename,
        "scores": {
            "correctness": avg_correctness_score,
            "answer_relevancy": avg_relevancy_score,
        },
        "reasons": conversation_reasons,
    }


def evaluate_deepeval_metrics(
    ground_truth_dir: str, predictions_dir: str, model: DeepEvalBaseLLM = None
) -> Dict[str, float]:
    """
    Evaluate DeepEval metrics for conversations in the given directories.

    Args:
        ground_truth_dir (str): Directory containing ground truth conversation JSON files.
        predictions_dir (str): Directory containing predicted conversation JSON files.
        model (DeepEvalBaseLLM, optional): The model to use for evaluation. Defaults to None.

    Returns:
        Dict[str, float]: A dictionary containing average metric scores and individual conversation scores.
    """
    filenames = [
        f
        for f in os.listdir(ground_truth_dir)
        if f.endswith(".json") and os.path.exists(os.path.join(predictions_dir, f))
    ]

    # Create a partial function with fixed arguments
    process_func = partial(
        process_conversation,
        ground_truth_dir=ground_truth_dir,
        predictions_dir=predictions_dir,
        model=model,
    )

    # Use multiprocessing to process conversations in parallel
    with multiprocessing.Pool() as pool:
        individual_scores = list(
            tqdm(pool.imap(process_func, filenames), total=len(filenames))
        )

    conversation_count = len(individual_scores)
    total_scores = {
        "correctness": sum(
            score["scores"]["correctness"] for score in individual_scores
        ),
        "answer_relevancy": sum(
            score["scores"]["answer_relevancy"] for score in individual_scores
        ),
    }

    # Calculate average correctness and relevancy scores across all conversations
    avg_correctness = (
        total_scores["correctness"] / conversation_count
        if conversation_count > 0
        else 0
    )
    avg_relevancy = (
        total_scores["answer_relevancy"] / conversation_count
        if conversation_count > 0
        else 0
    )

    return {
        "average_scores": {
            "correctness": avg_correctness,
            "answer_relevancy": avg_relevancy,
        },
        "individual_scores": individual_scores,
        "total_conversations": conversation_count,
    }
