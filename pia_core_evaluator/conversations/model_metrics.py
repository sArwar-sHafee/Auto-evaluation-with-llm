import os
import numpy as np
from typing import List
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _get_embeddings(texts):
    response = client.embeddings.create(model="text-embedding-ada-002", input=texts)
    return [np.array(item.embedding) for item in response.data]


def _cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )


def calculate_conversation_similarity(actual: List[str], predicted: List[str]) -> float:
    embeddings1 = _get_embeddings(actual)
    embeddings2 = _get_embeddings(predicted)

    total_score = 0.0
    count = 0

    for emb1 in embeddings1:
        for emb2 in embeddings2:
            total_score += _cosine_similarity(emb1, emb2)
            count += 1

    average_similarity = total_score / count if count != 0 else 0
    return average_similarity
