import unittest
from pia_core_evaluator.tools_calling.utils import (
    match_function,
    validate_function_flow,
    validate_function_accuracy,
    extract_functions_from_conversation,
)


class TestToolCalling(unittest.TestCase):
    def test_match_function_valid(self):
        ground_truth = {
            "function": {
                "name": "test_func",
                "arguments": {"arg1": "value1", "arg2": "value2"},
            }
        }
        predicted = {
            "function": {
                "name": "test_func",
                "arguments": {"arg1": "value1", "arg2": "value2"},
            }
        }
        result = match_function(ground_truth, predicted)
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_match_function_name_mismatch(self):
        ground_truth = {
            "function": {"name": "test_func", "arguments": {"arg1": "value1"}}
        }
        predicted = {
            "function": {"name": "wrong_func", "arguments": {"arg1": "value1"}}
        }
        result = match_function(ground_truth, predicted)
        self.assertFalse(result["valid"])
        self.assertIn("Function name mismatch", result["errors"][0])

    def test_match_function_missing_argument(self):
        ground_truth = {
            "function": {
                "name": "test_func",
                "arguments": {"arg1": "value1", "arg2": "value2"},
            }
        }
        predicted = {"function": {"name": "test_func", "arguments": {"arg1": "value1"}}}
        result = match_function(ground_truth, predicted)
        self.assertFalse(result["valid"])
        self.assertIn("Missing argument", result["errors"][0])

    def test_match_function_unexpected_argument(self):
        ground_truth = {
            "function": {"name": "test_func", "arguments": {"arg1": "value1"}}
        }
        predicted = {
            "function": {
                "name": "test_func",
                "arguments": {"arg1": "value1", "arg2": "value2"},
            }
        }
        result = match_function(ground_truth, predicted)
        self.assertFalse(result["valid"])
        self.assertIn("Unexpected argument", result["errors"][0])

    def test_validate_function_flow_valid(self):
        ground_truth = [
            {"function": {"name": "func1", "arguments": {"a": "1"}}},
            {"function": {"name": "func2", "arguments": {"b": "2"}}},
        ]
        predicted = [
            {"function": {"name": "func1", "arguments": {"a": "1"}}},
            {"function": {"name": "func2", "arguments": {"b": "2"}}},
        ]
        result = validate_function_flow(ground_truth, predicted)
        self.assertTrue(result["valid_flow"])
        self.assertIn("All functions in the flow are valid", result["reasons"][0])

    def test_validate_function_flow_invalid_count(self):
        ground_truth = [
            {"function": {"name": "func1", "arguments": {"a": "1"}}},
            {"function": {"name": "func2", "arguments": {"b": "2"}}},
        ]
        predicted = [{"function": {"name": "func1", "arguments": {"a": "1"}}}]
        result = validate_function_flow(ground_truth, predicted)
        self.assertFalse(result["valid_flow"])
        self.assertIn("Mismatch in number of functions", result["reasons"][0])

    def test_validate_function_flow_invalid_sequence(self):
        ground_truth = [
            {"function": {"name": "func1", "arguments": {"a": "1"}}},
            {"function": {"name": "func2", "arguments": {"b": "2"}}},
        ]
        predicted = [
            {"function": {"name": "func2", "arguments": {"b": "2"}}},
            {"function": {"name": "func1", "arguments": {"a": "1"}}},
        ]
        result = validate_function_flow(ground_truth, predicted)
        self.assertFalse(result["valid_flow"])
        self.assertIn("Function name mismatch at index 0", result["reasons"][0])

    def test_extract_functions_from_conversation_empty(self):
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]

        result = extract_functions_from_conversation(conversation)
        self.assertEqual(result, [])

    def test_extract_functions_from_conversation_no_router_tools(self):
        conversation = [
            {"role": "user", "content": "Hello"},
            {
                "role": "tool_call",
                "tool_calls": [
                    {
                        "function": {
                            "name": "tool_func1",
                            "arguments": {"arg1": "value1"},
                        }
                    },
                    {
                        "function": {
                            "name": "tool_func2",
                            "arguments": {"arg2": "value2"},
                        }
                    },
                ],
            },
        ]

        expected_output = [
            {"function": {"name": "tool_func1", "arguments": {"arg1": "value1"}}},
            {"function": {"name": "tool_func2", "arguments": {"arg2": "value2"}}},
        ]

        result = extract_functions_from_conversation(conversation)
        self.assertEqual(result, expected_output)

    def test_validate_function_accuracy(self):
        ground_truth_funcs = [
            {"function": {"name": "func1", "arguments": {"a": "1"}}},
            {"function": {"name": "func2", "arguments": {"b": "2"}}},
            {"function": {"name": "func3", "arguments": {"c": "3"}}},
        ]
        predicted_funcs = [
            {"function": {"name": "func1", "arguments": {"a": "1"}}},
            {"function": {"name": "func2", "arguments": {"b": "2"}}},
            {"function": {"name": "func4", "arguments": {"d": "4"}}},
        ]
        result = validate_function_accuracy(ground_truth_funcs, predicted_funcs)
        self.assertEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["valid_matches"], 2)
        self.assertEqual(result["total_functions"], 3)
        self.assertEqual(len(result["function_validations"]), 3)
        self.assertTrue(result["function_validations"][0]["valid"])
        self.assertTrue(result["function_validations"][1]["valid"])
        self.assertFalse(result["function_validations"][2]["valid"])


if __name__ == "__main__":
    unittest.main()
