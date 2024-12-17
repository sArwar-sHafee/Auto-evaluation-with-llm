import os
import json
import random
import argparse
from autogen import Cache
from autogen import OpenAIWrapper
from tqdm import tqdm
import scenario_gen_config as scenario_cfg


# Set up environment variables and constants
os.environ["OPENAI_API_KEY"] = scenario_cfg.API_KEY

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dump_dir", type=str, default="data/user_scenarios/weather_assistant"
)
parser.add_argument("--num_scenarios", type=int, default=1)
args = parser.parse_args()


def generate_user_scenario_prompt():
    return f"""You are an expert trainer for {scenario_cfg.ASSISTANT_FOR} call center agents. You work for the {scenario_cfg.ASSISTANT_FOR} call center. People call the {scenario_cfg.SHORT_FORM_OF_ASSISTANT_FOR} helpline for various reasons, such as to {scenario_cfg.CALL_REASON}. You are responsible for training new agents to handle these calls. You have been asked to create a training scenario for new agents. Write 30 user scenarios that will help new agents practice their skills. The scenario should be realistic and should help agents learn how to handle calls effectively. The scenario should include a description of the situation and the caller's problem. The scenario should be detailed.

You can pick any reason for call, but you can focus on scenarios related to {scenario_cfg.CALL_REASON}. You can also include other scenarios that are common in Bangladesh. The caller will be Bangladeshi. The scenario should be detailed and should include multiple reasons for a single call to the call center.

Also sometime add that caller wants to get email or sms or book appointment schedule from the call center.

YOU MUST INCLUDE MULTIPLE REASONS FOR A SINGLE CALL TO THE CALL CENTER. THE CALLERS MUST BE BANGLADESHI.

NOW GENERATE 30 USER SCENARIOS FOR CALL CENTER AGENTS. YOU MUST KEEP BANGLADESHI CONTEXT IN MIND.
YOUR OUTPUT MUST BE A JSON LIST CONTAINING {scenario_cfg.NUMBER_OF_SCENARIOS} USER SCENARIOS.
"""


def generate_scenarios():
    client = OpenAIWrapper()
    cache = Cache.disk()
    os.makedirs(args.dump_dir, exist_ok=True)

    prompt = generate_user_scenario_prompt()

    for idx in tqdm(range(args.num_scenarios)):
        messages = [{"role": "system", "content": prompt}]
        temperature = random.uniform(0.3, 0.99)

        while True:
            try:
                response = client.create(
                    model=scenario_cfg.MODEL_NAME,
                    cache=cache,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )

                str_response = client.extract_text_or_completion_object(response)[0]
                json_response = json.loads(str_response)

                with open(f"{args.dump_dir}/{idx}.json", "w") as f:
                    json.dump(json_response, f, ensure_ascii=False, indent=4)
                break  # Successfully generated the data
            except Exception as e:
                print(f"Error occurred: {e}")
                temperature += 0.05


if __name__ == "__main__":
    generate_scenarios()
