import json
import traceback
from typing import List
from loguru import logger
from openai import OpenAI


def extract_functions_from_conversation(conversation: List[dict]) -> List[dict]:
    """
    Extract functions from the conversation and create a function list.
    Router tools are placed before tool_calls in the list.

    Args:
        conversation (List[dict]): A list of conversation dictionaries.

    Returns:
        List[dict]: A list of extracted functions and router tools.
    """
    function_list = []

    for entry in conversation:
        if entry.get("role") == "tool_call":
            function_list.extend(entry["tool_calls"])

    return function_list


def match_function(
    ground_truth_func: dict, predicted_func: dict, match_arg_value: bool = True
) -> dict:
    """
    Match two functions by their name and arguments and return a dictionary with validation results.

    Args:
        ground_truth_func (dict): The ground truth function dictionary.
        predicted_func (dict): The predicted function dictionary.

    Returns:
        dict: A dictionary containing validation results.
    """
    result = {"valid": True, "errors": []}

    try:
        # Check function name
        if ground_truth_func["function"]["name"] != predicted_func["function"]["name"]:
            result["valid"] = False
            result["errors"].append(
                f"Function name mismatch: expected '{ground_truth_func['function']['name']}', got '{predicted_func['function']['name']}'"
            )

        # Check arguments
        gt_args = ground_truth_func["function"]["arguments"]
        pred_args = predicted_func["function"]["arguments"]
        logger.debug(f"gt_args: {gt_args}")
        logger.debug(f"pred_args: {pred_args}")

        for key in gt_args:
            if key not in pred_args:
                result["valid"] = False
                result["errors"].append(f"Missing argument: '{key}'")
            elif match_arg_value and (
                str(gt_args[key]).lower() not in str(pred_args[key]).lower()
            ):
                result["valid"] = False
                result["errors"].append(
                    f"Argument value mismatch for '{key}': expected '{gt_args[key]}', got '{pred_args[key]}'"
                )

        for key in pred_args:
            if key not in gt_args:
                result["valid"] = False
                result["errors"].append(f"Unexpected argument: '{key}'")
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Error matching functions: {str(e)}")
        logger.error(traceback.format_exc())

    return result


def validate_function_flow(
    ground_truth_funcs: List[dict],
    predicted_funcs: List[dict],
    match_arg_value: bool = True,
) -> dict:
    """
    Validate the flow of functions between ground truth and predictions.

    Args:
        ground_truth_funcs (list): List of ground truth function dictionaries.
        predicted_funcs (list): List of predicted function dictionaries.

    Returns:
        dict: A dictionary containing flow validation results.
    """
    result = {"valid_flow": True, "reasons": [], "function_validations": []}

    if len(ground_truth_funcs) != len(predicted_funcs):
        result["valid_flow"] = False
        result["reasons"].append(
            f"Mismatch in number of functions: expected {len(ground_truth_funcs)}, got {len(predicted_funcs)}"
        )
        return result

    for i, (gt_func, pred_func) in enumerate(zip(ground_truth_funcs, predicted_funcs)):
        # Check function name matching by index
        if gt_func["function"]["name"] != pred_func["function"]["name"]:
            result["valid_flow"] = False
            result["reasons"].append(
                f"Function name mismatch at index {i}: expected '{gt_func['function']['name']}', got '{pred_func['function']['name']}'"
            )

        # Check function validity using the match_function function
        func_validation = match_function(
            gt_func, pred_func, match_arg_value=match_arg_value
        )
        func_validation["function_id"] = i  # Add function index as function_id
        result["function_validations"].append(func_validation)

        if not func_validation["valid"]:
            result["valid_flow"] = False
            result["reasons"].append(f"Function at index {i} is invalid")
            result["reasons"].extend(func_validation["errors"])

    if result["valid_flow"]:
        result["reasons"].append(
            "All functions in the flow are valid and match the expected sequence"
        )

    return result


def validate_function_accuracy(
    ground_truth_funcs: List[dict],
    predicted_funcs: List[dict],
    match_arg_value: bool = True,
) -> dict:
    """
    Validate the accuracy of predicted functions against ground truth functions.

    Args:
        ground_truth_funcs (List[dict]): List of ground truth function dictionaries.
        predicted_funcs (List[dict]): List of predicted function dictionaries.

    Returns:
        dict: A dictionary containing accuracy and validation results.
    """
    result = {
        "accuracy": 0.0,
        "valid_matches": 0,
        "total_functions": len(ground_truth_funcs),
        "function_validations": [],
    }

    for i, gt_func in enumerate(ground_truth_funcs):
        matched = False
        for pred_func in predicted_funcs:
            if gt_func["function"]["name"] == pred_func["function"]["name"]:
                func_validation = match_function(
                    gt_func, pred_func, match_arg_value=match_arg_value
                )
                func_validation["function_id"] = i
                result["function_validations"].append(func_validation)

                if func_validation["valid"]:
                    result["valid_matches"] += 1
                    matched = True
                break

        if not matched:
            result["function_validations"].append(
                {
                    "function_id": i,
                    "valid": False,
                    "errors": [
                        f"No matching function found for '{gt_func['function']['name']}'"
                    ],
                }
            )

    result["accuracy"] = (
        result["valid_matches"] / result["total_functions"]
        if result["total_functions"] > 0
        else 0.0
    )

    return result


def match_function_llm(
    ground_truth_func: dict, predicted_func: dict, model: str = "gpt-4o"
) -> dict:
    """
    Match two functions using an LLM.

    Args:
        ground_truth_func (dict): The ground truth function dictionary.
        predicted_func (dict): The predicted function dictionary.

    Returns:
        dict: A dictionary containing validation results.
    """
    client = OpenAI()

    try:
        system_prompt = {
            "role": "system",
            "content": 'Compare two given functions (ground truth and prediction) and determine if they match. Return the result in a specified format.\n\n- The input consists of two JSON objects representing the ground truth function and the prediction function.\n- Compare these objects based on their structure and content.\n- Return a JSON indicating whether the prediction matches the ground truth and if there are any errors.\n\n# Steps\n\n1. **Parse Input**: Extract the two JSON objects provided, identified as the ground truth function and prediction function.\n2. **Compare Attributes**: Check that:\n   - The "type" field matches in both functions.\n   - The "function" names are the same.\n   - All "arguments" match, accounting for equivalent representations (e.g., transliteration, synonyms, translated).\n3. **Determine Validity**: Establish if the prediction function is an accurate representation of the ground truth function.\n4. **Compile Errors**: If discrepancies are found, document them in an error list. Each error should specify the attribute and describe the difference.\n\n# Output Format\n\n- Dictionary object with no markdown:\n  - `valid`: A boolean indicating if the prediction matches the ground truth.\n  - `errors`: A list detailing any inconsistencies found. If no errors are found, this should be an empty list.\n\n# Example\n\n**Input:**\n\nground_truth_function:\n```json\n{\n  "id": null,\n  "type": "function",\n  "function": {\n    "name": "weather_check",\n    "arguments": {\n      "q": "Dhaka"\n    }\n  }\n}\n```\n\nprediction_function:\n```json\n{\n  "id": null,\n  "type": "function",\n  "function": {\n    "name": "weather_check",\n    "arguments": {\n      "q": "ঢাকা"\n    }\n  }\n}\n```\n\n**Output:**\n\n{"valid": true, "errors": []}\n\n# Notes\n\n- Consider different ways to represent the same argument (e.g., transliterations, synonyms, translated).\n- Ensure that the attributes being compared are within the nested structure of JSON objects.\n- If multiple errors are present, list them all in the errors array.\n- Always return the result without any explanation, only the dictionary',
        }

        input_prompt = {
            "role": "user",
            "content": f"ground_truth_function: {ground_truth_func}\n\nprediction_function: {predicted_func}",
        }
        messages = [system_prompt, input_prompt]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "text"},
        )

        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error matching functions: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "valid": False,
            "errors": [f"Error matching functions using LLM: {str(e)}"],
        }
