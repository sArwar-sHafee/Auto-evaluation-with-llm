# PIA LLM Evaluation Framework
PIA LLM Evaluation framework is dedicated to evaluate the performance of LLMs in the context of PIA framework. This framework is designed to be flexible and extensible to support various evaluation tasks and models. We have several evaluation tasks including:
1. Tool Calling Evaluation
    i. Tool Flow Evaluation to check if the tool call happen in correct order.
    ii. Tool validation to check if the called tool is correct or wrong.
2. Conversation Evaluation
    i. Heuristic Conversation Evaluation using different heuristic Evaluation method like BLEU, ROUGE etc.
    ii. Model-based conversation evaluation like semantic similarity finding using embedding model.
    iii. DeepEval-based correctness and relevancy evaluation using LLM as a judge.
3. Benchmarking
    - Benchmark the results to generate all the evaluation with summaries


## Environment Setup
To setup the environment simply install the dependencies by

```
pip install -r requirements.txt
```

## Benchmarking
### Automatic Benchmarking
Change environment path in [.github/workflows/automatic_pia_benchmark.yml](.github/workflows/automatic_pia_benchmark.yml)

It will automatically run the benchmark on the latest commit and push the results to the [benchmarks](benchmarks) folder.

### Manual Benchmarking
- To generate PIA data do the following command
```bash
python pia_data_generator.py \
--conversation_jsons_path ../benchmarks/bench_data/weather_assistant/gt_data \
--graph_json_path ../benchmarks/bench_data/weather_assistant/ws_graph_data.json \
--dump_path ../benchmarks/bench_data/weather_assistant/temp \
--base_url http://34.41.217.187:8000/api

```
- To generate the benchmark results do the following command

```bash
cd scripts
# setup the openai api key before running the benchmark
# export OPENAI_API_KEY="api key"
python benchmark.py \
--tool_calling \
--heuristic \
--model \
--deepeval \
--match_function_arg_value \
--ground_truth_file_path gt_conversation_path/my_assistant \
--predictions_file_path pia_llm_generated_conversation_path/my_assistant \
--dump_path mypath/benchmark_gpt4o/my_assistant
```


## Data Generator
This module also contains ground truth data generation, pia data generations.

Follow [Data Generation README](data_generator/README.md) for more details.


## Eval benchmark directory organization

```
benchmarks/bench_data
    - sales_promotion
        - sales_1
            - gt_data
            - pia_data
            graph_data.json
        - sales_2
            -gt_data
            - pia_data
            graph_data.json
```

- checnage in action `automatic_pia_benchmark.yml`

```
CONVERSATION_JSONS_PATH: '../benchmarks/bench_data/weather_assistant/gt_data'
GRAPH_JSON_PATH: '../benchmarks/bench_data/weather_assistant/ws_graph_data.json'
PIA_DUMP_PATH: '../benchmarks/bench_data/weather_assistant/pia_data/gpt4omini_test'
BENCHMARK_RESULTS_PATH: '../benchmarks/results/weather_assistant/gpt4omini'
```
