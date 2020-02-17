#!/bin/bash

cd ../process/ ; \

python3 process_wrangled.py ; \

python3 process_wrangled.py --max_authors 20 --min_authors 10 --train_test_ratio 0.7 ; \

python3 create_embedding_tokens.py ; \

python3 create_tokens.py --train_data_file top20_10_train.json --test_data_file top20_10_test.json

cd ../src/ ; \

python3 train.py
