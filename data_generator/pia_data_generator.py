import os
import glob
import json
import uuid
import argparse
from tqdm import tqdm
from loguru import logger
from typing import List
from pia_llm import get_pia_response

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conversation_jsons_path",
    type=str,
    default="data/gt_data/gpt4o/weather_assistant",
)
parser.add_argument(
    "--graph_json_path", type=str, default="data/graph_samples/weather_assistant.json"
)
parser.add_argument(
    "--dump_path", type=str, default="data/pia_data/gpt4o/weather_assistant"
)
parser.add_argument("--base_url", type=str, default="http://34.44.20.192:5077/api")
args = parser.parse_args()


def generator(
    base_url: str, session_id: str, graph_data: dict, gt_messages: List[dict]
):
    """
    Generate PIA data from the given GT messages.

    Args:
        base_url (str): The base URL for the PIA API.
        session_id (str): The session ID for the PIA API.
        graph_data (dict): The graph data for the PIA API.
        gt_messages (list): List of messages from the GT. each message is a dict with 'role' and 'content' keys.
    """
    generated_data = []

    for message in tqdm(gt_messages):
        if message["role"] == "user":
            user_message = message["content"]

            logger.info(f"User message: {user_message}")
            generated_data.append({"role": "user", "content": user_message})

            pia_response = get_pia_response(
                base_url, session_id, user_message, graph_data
            )
            logger.info(f"Pia response: {pia_response}")
            if pia_response["role"] == "assistant":
                routing_tools = pia_response.get("routing_tools", [])
                functional_tools = pia_response.get("functional_tools", [])
                for tool in routing_tools:
                    generated_data.append(tool)
                for tool in functional_tools:
                    generated_data.append(tool)

                content = pia_response["content"]
                generated_data.append({"role": "assistant", "content": content})
            # elif pia_response["role"] == "tool_call":
            #     content = pia_response["content"]
            #     # add the tool call to the generated data
            #     generated_data.append(pia_response)
            #     # add the assistant response to the generated data
            #     generated_data.append({"role": "assistant", "content": content})

            logger.info(f"Generated data: {generated_data}")

    return generated_data


def generate_single_conversation(
    conversation_json_path: str,
    graph_json_path: str,
    dump_path: str,
    base_url: str,
    session_id: str,
):

    os.makedirs(dump_path, exist_ok=True)

    with open(conversation_json_path) as file:
        gt_messages = json.load(file)
    with open(graph_json_path) as file:
        graph_data = json.load(file)
    generated_data = generator(base_url, session_id, graph_data, gt_messages)

    dump_json_file = os.path.join(dump_path, os.path.basename(conversation_json_path))
    with open(dump_json_file, "w") as file:
        json.dump(generated_data, file, indent=4, ensure_ascii=False)


def main():
    conversation_jsons = glob.glob(args.conversation_jsons_path + "/*.json")
    logger.info(f"Found {len(conversation_jsons)} conversation jsons")

    for conversation_json_path in tqdm(conversation_jsons):
        conversation_dump_path = os.path.join(
            args.dump_path, os.path.basename(conversation_json_path)
        )
        if os.path.exists(conversation_dump_path):
            continue
        session_id = str(uuid.uuid4())
        generate_single_conversation(
            conversation_json_path,
            args.graph_json_path,
            args.dump_path,
            args.base_url,
            session_id,
        )
        logger.info(f"Generated conversation for {conversation_json_path}")


if __name__ == "__main__":
    main()
