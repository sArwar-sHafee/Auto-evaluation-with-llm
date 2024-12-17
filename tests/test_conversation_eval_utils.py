import unittest
from pia_core_evaluator.conversations.utils import extract_messages_from_conversation


class TestExtractMessagesFromConversation(unittest.TestCase):
    def test_extract_user_messages(self):
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "Goodbye"},
        ]
        expected_output = ["Hello", "How are you?", "Goodbye"]
        self.assertEqual(
            extract_messages_from_conversation(conversation), expected_output
        )

    def test_extract_assistant_messages(self):
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "Goodbye"},
        ]
        expected_output = ["Hi there!", "I'm doing well, thank you!"]
        self.assertEqual(
            extract_messages_from_conversation(conversation, role="assistant"),
            expected_output,
        )

    def test_empty_conversation(self):
        conversation = []
        self.assertEqual(extract_messages_from_conversation(conversation), [])

    def test_no_matching_role(self):
        conversation = [
            {"role": "system", "content": "This is a system message"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        self.assertEqual(extract_messages_from_conversation(conversation), [])


if __name__ == "__main__":
    unittest.main()
