from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM


def get_correctness_metric(model: DeepEvalBaseLLM = None):
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        # NOTE: you can only provide either criteria or evaluation_steps, and not both
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
    )
    return correctness_metric


def get_answer_relevancy_metric(model: DeepEvalBaseLLM = None, threshold: float = 0.7):
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=threshold, model=model, include_reason=True
    )
    return answer_relevancy_metric
