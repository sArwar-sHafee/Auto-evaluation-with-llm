import requests
from loguru import logger

# url = "http://127.0.0.1:8000/api"
# url = "http://34.44.20.192:5077/api"


def get_pia_response(base_url, session_id, user_message, graph_data):
    logger.info(f"Getting PIA response for user message: {user_message}")
    payload = {
        "session_id": session_id,
        "user_message": user_message,
        "graph_data": graph_data,
    }
    # Define headers
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    response = requests.post(base_url, headers=headers, json=payload)
    if response.status_code == 200:
        logger.info(f"Pia response: {response.json()}")
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}")
