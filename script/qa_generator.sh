export OPENAI_API_KEY="OPENAI_API_KEY"

python3 qa_generator.py \
--graph_data_path ../graph_data/bright_dental_appointment.json \
--output_path ../qa_pairs/bright_dental_appointment/ \
--quantity_of_categories 10 \
--qa_pairs_per_category 10
