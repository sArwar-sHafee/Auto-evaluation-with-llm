import sys
sys.path.append("..")
import os
from generator.questionnaire import generate_qa_pairs
from generator.catergory_gen import category_generator
import json
import argparse
from loguru import logger


parser = argparse.ArgumentParser()
parser.add_argument("--graph_data_path", type=str, default="graph_data/bright_dental_appointment.json")
parser.add_argument("--output_path", type=str, default="qa_pairs/bright_dental_appointment")
parser.add_argument("--quantity_of_categories", type=int, default=10)
parser.add_argument("--qa_pairs_per_category", type=int, default=10)

args = parser.parse_args()

graph_data_path = args.graph_data_path
output_path = args.output_path
quantity_of_categories = args.quantity_of_categories
qa_pairs_per_category = args.qa_pairs_per_category

def save_qa_pairs(qa_pairs, output_path, category_name):
    """
    Save the QA pairs to the given output path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_dir = os.path.dirname(output_path)
    file_index = 0
    while os.path.exists(os.path.join(base_dir, f"{category_name}_{file_index}.json")):
        file_index += 1
    output_file_path = os.path.join(base_dir, f"{category_name}_{file_index}.json")
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    with open(graph_data_path, "r") as f:
       graph_data = json.load(f)
    logger.info(f"Loaded graph data from {graph_data_path}")
    if "full_prompt" in graph_data["nodes"][0]:
        prompt = graph_data["nodes"][0]["full_prompt"]
        tools = graph_data["nodes"][0]["tools"]
    elif "prompt" in graph_data["nodes"][0]:
        prompt = graph_data["nodes"][0]["prompt"]
        if "prompt_info" in graph_data:
            prompt = str(graph_data["prompt_info"]) + "\n" + str(prompt)
        tools = graph_data["nodes"][0]["tools"] + graph_data["nodes"][0]["edges"]
    else:
        raise ValueError("Graph data does not contain valid prompt or full_prompt")
    categories = category_generator(prompt, tools, quantity_of_categories)
    logger.info(f"Generated {quantity_of_categories} categories")
    for category in categories:
        qa_pairs = generate_qa_pairs(prompt, tools, category, qa_pairs_per_category)
        save_qa_pairs(qa_pairs, output_path, category["topic_name"])
        logger.info(f"Saved QA pairs to {output_path}")