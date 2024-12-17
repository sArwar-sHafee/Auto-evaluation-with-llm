import json
from user_scenario_generator import generate_scenarios
from gt_data_generator import generate_gt_data
from pia_data_generator import generate_pia_data


def test_generate_user_scenerio():
    user_scenario = generate_scenarios()
    with open("./data/user_scenarios/0.json", "w") as file:
        json.dump(user_scenario, file, ensure_ascii=False)


def test_gt_data_generator():
    with open("./data/user_scenarios/0.json") as file:
        scenarios = json.load(file)
    scenario = str(scenarios["scenarios"][1])

    base_url = "http://34.44.20.192:5077/api"
    session_id = "1"
    with open("./data/graph_samples/test_1.json") as file:
        graph_data = json.load(file)

    init_user_message = "হ্যালো"
    generated_data = generate_gt_data(
        base_url, session_id, graph_data, scenario, init_user_message
    )
    with open("./data/sample_data.json", "w") as file:
        json.dump(generated_data, file, ensure_ascii=False)


def test_pia_data_generator():
    with open("./data/sample_data.json") as file:
        gt_messages = json.load(file)
    with open("./data/graph_samples/test_1.json") as file:
        graph_data = json.load(file)
    base_url = "http://34.44.20.192:5077/api"
    session_id = "100"
    generated_data = generate_pia_data(base_url, session_id, graph_data, gt_messages)
    with open("./data/sample_data_pia.json", "w") as file:
        json.dump(generated_data, file, ensure_ascii=False)


if __name__ == "__main__":
    test_gt_data_generator()
    # test_pia_data_generator()
