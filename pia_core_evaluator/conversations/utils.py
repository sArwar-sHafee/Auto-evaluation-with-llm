from typing import List


def extract_messages_from_conversation(
    conversation: List[dict], role: str = "user"
) -> List[str]:
    """
    Extract messages from the conversation and create a message list.

    Args:
        conversation (List[dict]): A list of conversation dictionaries.

    Returns:
        List[str]: A list of extracted messages.
    """
    message_list = []

    for entry in conversation:
        if entry.get("role") == role:
            message_list.append(entry.get("content"))

    return message_list
