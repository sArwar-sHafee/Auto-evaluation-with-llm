import evaluate
import numpy as np
from loguru import logger
from typing import Callable, List, Tuple
from rouge_score import rouge_scorer
from bnlp import BasicTokenizer


def calculate_rouge(
    actual_list, predicted_list, tokenizer: Callable = BasicTokenizer()
) -> Tuple[float, float]:
    # Check if tokenizer has tokenize method, otherwise use default BasicTokenizer
    if not hasattr(tokenizer, "tokenize"):
        tokenizer = BasicTokenizer()
        logger.warning(
            "Provided tokenizer does not have 'tokenize' method. Using BasicTokenizer instead."
        )

    # Create a RougeScorer with custom tokenizer and without stemming
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"], use_stemmer=False, tokenizer=tokenizer
    )
    rouge_scores = [scorer.score(a, p) for a, p in zip(actual_list, predicted_list)]
    avg_rouge1 = np.mean([score["rouge1"].fmeasure for score in rouge_scores])
    avg_rougeL = np.mean([score["rougeL"].fmeasure for score in rouge_scores])
    return avg_rouge1, avg_rougeL


class Heuristic:
    def __init__(self, tokenizer: Callable = None):
        self.tokenizer = tokenizer
        self.bleu = evaluate.load("bleu", tokenizer=tokenizer)
        self.sacrebleu = evaluate.load("sacrebleu", tokenizer=tokenizer)
        self.meteor = evaluate.load("meteor", tokenizer=tokenizer)

    def __call__(self, references: List[str], candidates: List[str]) -> dict:
        bleu = self.bleu.compute(
            predictions=candidates, references=references, max_order=1
        )
        sacrebleu = self.sacrebleu.compute(
            predictions=candidates, references=references
        )
        rouge = calculate_rouge(references, candidates, self.tokenizer)
        meteor = self.meteor.compute(predictions=candidates, references=references)

        return {
            "bleu": bleu.get("bleu", 0),
            "sacrebleu": sacrebleu.get("score", 0),
            "rouge1": rouge[0],
            "rougeL": rouge[1],
            "meteor": meteor.get("meteor", 0),
        }
