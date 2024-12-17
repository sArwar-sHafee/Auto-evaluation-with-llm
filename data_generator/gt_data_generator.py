import uuid
import os
import json
import glob
import argparse
from tqdm import tqdm
from typing import List, Optional
from loguru import logger
from user_prompt import human_caller_prompt
from auto_user import get_user_response
from pia_llm import get_pia_response

parser = argparse.ArgumentParser()
parser.add_argument("--base_url", type=str, default="http://34.44.20.192:5077/api")
parser.add_argument(
    "--graph_json_path", type=str, default="data/graph_samples/weather_assistant.json"
)
parser.add_argument(
    "--dump_path", type=str, default="data/gt_data/gpt4o/weather_assistant"
)
parser.add_argument(
    "--scenarios_path", type=str, default="data/user_scenarios/weather_assistant"
)
parser.add_argument("--init_user_message", type=str, default="হ্যালো")

args = parser.parse_args()


def generator(
    base_url: str,
    session_id: str,
    graph_data: dict,
    scenerio: Optional[str],
    init_user_message: str = "hello",
):
    if not isinstance(scenerio, str):
        scenerio = str(scenerio)

    user_input_messages = [
        {
            "role": "system",
            "content": human_caller_prompt.format(user_scenario=scenerio),
        },
        {"role": "user", "content": init_user_message},
    ]

    generated_data = []

    while True:
        user_message = get_user_response(user_input_messages, init_user_message)
        logger.info(f"User message: {user_message}")
        if "<human_caller_ends_the_call>" in user_message:
            break
        user_input_messages.append({"role": "assistant", "content": user_message})
        generated_data.append({"role": "user", "content": user_message})

        pia_response = get_pia_response(base_url, session_id, user_message, graph_data)
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

            if (
                len(functional_tools) > 0
                and functional_tools[-1]["tool_calls"][0]["function"]["name"]
                == "end_call"
            ):
                break

        logger.info(f"Generated data: {generated_data}")

    return generated_data


def generate_multiple_conversations(
    base_url: str,
    graph_data: dict,
    scenarios: List[str],
    dump_dir: str,
    init_user_message: str = "hello",
):
    os.makedirs(dump_dir, exist_ok=True)

    for i, scenario in tqdm(enumerate(scenarios)):
        session_id = str(uuid.uuid4())
        conversation_data = generator(
            base_url, session_id, graph_data, scenario, init_user_message
        )

        filename = f"conversation_{i+1}_{session_id}.json"
        filepath = os.path.join(dump_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Generated conversation for scenario {i+1} and saved to {filepath}"
        )


def main():
    scenario_json_path = args.scenarios_path
    graph_json_path = args.graph_json_path
    dump_path = args.dump_path
    scenario_jsons = glob.glob(scenario_json_path + "/*.json")
    logger.info(f"Found {len(scenario_jsons)} scenario jsons")

    with open(graph_json_path, "r") as f:
        graph_data = json.load(f)

    for scenario_json_path in tqdm(scenario_jsons):

        with open(scenario_json_path, "r") as f:
            scenarios = json.load(f).get("user_scenarios", [])
            if not scenarios:
                raise ValueError(f"No scenarios found in {scenario_json_path}")

        generate_multiple_conversations(
            args.base_url,
            graph_data,
            scenarios,
            dump_path,
            init_user_message=args.init_user_message,
        )


if __name__ == "__main__":
    main()
