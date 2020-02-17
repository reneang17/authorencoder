#!/bin/bash

python process_wrangled.py ; \ 

python process_wrangled.py --max_authors 20 --min_authors 10 --train_test_ratio 0.7 ; \

python create_embedding_tokens.py ; \

python create_tokens.py --train_data_file top20_10_train.json --test_data_file top20_10_test.json
