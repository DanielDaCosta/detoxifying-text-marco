python3 rewrite_together.py --data_path datasets/sbf/sbfdev.csv --output_dir datasets/rewrites/rewritten_sbfdev/ --no_check_cost
python3 rewrite_together.py --data_path datasets/sbf/sbftst.csv --output_dir datasets/rewrites/rewritten_sbftst/ --no_check_cost
python3 rewrite_together.py --data_path datasets/dynabench/db_dev.csv --output_dir datasets/rewrites/rewritten_db_dev/ --no_check_cost
python3 rewrite_together.py --data_path datasets/dynabench/db_test.csv --output_dir datasets/rewrites/rewritten_db_test/ --no_check_cost
python3 rewrite_together.py --data_path datasets/microagressions/val.csv --output_dir datasets/rewrites/rewritten_microagressions_val/ --no_check_cost
python3 rewrite_together.py --data_path datasets/microagressions/test.csv --output_dir datasets/rewrites/rewritten_microagressions_test/ --no_check_cost