import sys

sys.path.append("../")
import os


from pia_core_evaluator.tools_calling.utils import match_function_llm


def test_function_match_llm():
    ground_truth_func = {
        "type": "function",
        "function": {"name": "weather_check", "arguments": {"q": "Dhaka"}},
    }
    predicted_func = {
        "type": "function",
        "function": {"name": "weather_check", "arguments": {"q": "ঢাকা"}},
    }

    print(match_function_llm(ground_truth_func, predicted_func))


if __name__ == "__main__":
    test_function_match_llm()
