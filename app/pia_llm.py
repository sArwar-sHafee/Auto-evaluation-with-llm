import requests
from loguru import logger

import asyncio

from eval_server.eval_processor import EvalProcessor
eval_processor = EvalProcessor()

from app.graph_loader import load_graph_data

# Remove the default console logger
logger.remove()

# Configure logger to write to a file
logger.add("logs/pia_logs.log", rotation="1 MB")


async def get_pia_response(session_id, user_message, graph_data_path):
    graph_data = load_graph_data(graph_data_path)
    response = await eval_processor.process_events(
            user_message, session_id, graph_data
        )
    if response["role"] == "assistant":
        return response["content"], response["routing_tools"] + response["functional_tools"]
    else:
        raise Exception(f"Request failed with status code {response.status_code}")
