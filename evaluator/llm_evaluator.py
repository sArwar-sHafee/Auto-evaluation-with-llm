from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM

def get_correctness_metric(model: DeepEvalBaseLLM = None):
    """
    Evaluate the response of the agent.
    """
    correctness_metric = GEval(
        name="Correctness",
        # criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK"
        ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=model
    )
    return correctness_metric

def correctness(user_message, ground_truth, agent_reply):
    correctness_metric = get_correctness_metric()
    if ground_truth is None:
        ground_truth = ""
    test_case = LLMTestCase(
            input=user_message, actual_output=agent_reply, expected_output=ground_truth
        )
    correctness_metric.measure(test_case)
    return correctness_metric.score