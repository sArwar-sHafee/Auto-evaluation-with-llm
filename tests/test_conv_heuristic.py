import unittest
from pia_core_evaluator.conversations.heuristic_mertics import Heuristic
from bnlp import BasicTokenizer


class TestHeuristic(unittest.TestCase):
    def setUp(self):
        self.heuristic = Heuristic()
        self.heuristic_with_tokenizer = Heuristic(tokenizer=BasicTokenizer())

    def test_heuristic_without_tokenizer(self):
        references = ["The cat is on the mat.", "The dog is in the yard."]
        candidates = ["A cat sits on a mat.", "There is a dog in the garden."]

        results = self.heuristic(references, candidates)

        self.assertIn("bleu", results)
        self.assertIn("sacrebleu", results)
        self.assertIn("rouge1", results)
        self.assertIn("rougeL", results)
        self.assertIn("meteor", results)

        self.assertIsInstance(results["bleu"], float)
        self.assertIsInstance(results["sacrebleu"], float)
        self.assertIsInstance(results["rouge1"], float)
        self.assertIsInstance(results["rougeL"], float)
        self.assertIsInstance(results["meteor"], float)

    def test_heuristic_with_tokenizer(self):
        references = ["The cat is on the mat.", "The dog is in the yard."]
        candidates = ["A cat sits on a mat.", "There is a dog in the garden."]

        results = self.heuristic_with_tokenizer(references, candidates)

        self.assertIn("bleu", results)
        self.assertIn("sacrebleu", results)
        self.assertIn("rouge1", results)
        self.assertIn("rougeL", results)
        self.assertIn("meteor", results)

        self.assertIsInstance(results["bleu"], float)
        self.assertIsInstance(results["sacrebleu"], float)
        self.assertIsInstance(results["rouge1"], float)
        self.assertIsInstance(results["rougeL"], float)
        self.assertIsInstance(results["meteor"], float)


if __name__ == "__main__":
    unittest.main()
