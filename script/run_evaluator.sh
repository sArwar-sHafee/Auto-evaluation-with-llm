export OPENAI_API_KEY="OPENAI_API_KEY"

python3 run_evaluator.py \
--qa_pairs_path ../qa_pairs/bright_dental \
--graph_data_path ../graph_data/bright_dental_appointment.json \
--output_path logs/results.csv
