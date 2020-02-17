
![library.](./media/library_panoramic.jpg)
# AuthorEncoder

Find poetry authors who writes similar to your dearest poems.

Would it be great if,  after reading a touching poem, we could find authors who have similar ideas or style? To this end, I build a scalable recommendation system that takes  as input a poem and outputs authors that write similarly. Bookshops can scale this service to empower their customers, boosting sales and loyalty. The underlying AI results from applying computer vision techniques  to encode the legacy of a hundred of classic and modern poets. 

## How does it work?

Tipically, recomendation systems are based on what other people like. Instead, AuthorEncoder makes recomendations based on 
on the poems themselfs. How is this possible? you might be asking. The answer is bringing the triple tecnique of [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) to natural language processing. Precisely, we wrangle and process poems, which are them used to build an authorencoder that multiple authors, which in turn can be used to give recomendation once the embedding itself. Run the main file to build this project yourself. 

## AuthorEncoder in action

![](./media/how_it_works.gif)

## About used data

The data to training this model can be found on [John Hallman repo](https://github.com/johnhallman/poem-data-processing) which to my understanding contains the larger [poetryfoundation.org](poetryfoundation.org). The required wrangling can be found on 
wrangling and EDA. 

## Requirements 

Appart from the requirements.txt, follow the instructions on [here](https://spacy.io/usage)
to install spaCy. 

## Current results and metrics

The present encoder is build with the 10 authors which produces the larges corpus of poems. 

---
## Parameter

processing_wrangle.py takes clean data and 

| Parameter  | Description | Default |
| ------------- | ------------- | ------------- |

|--data_dir | dir of wrangled data to process|../data/wrangled/ | 

|--data_file | wrangled data-file to process | wrangled_data.csv | 

|--process_dir | dir to dump processed data | ../data/processed/| 

|--split_into | max words per poem extract |100 |

|--min_words_per_author | Author min corpus word length | 2500 |

|--chars_to_keep | chars to keep  | \n?! |

|--no_white_space | Remove multiple white space | True |

|--no_newlines | bool | Remove multiple new lines | True |

|--max_authors | Keep authors with larges corpus | 10 |

|--min_authors | Larges corpus starting from author | 0 |

|--train_test_ratio | Train test ratio | 0.9 |

|--seed | Seed for replicability | 1234 |


Create word embedding, dictionary and tokenize poems

| Parameter  | Description | Default |
| ------------- | ------------- | ------------- |

| --data_dir | dir of data | ../data/wrangled/ |

| --train_data_file | train data file | top10_train.json | 

| --test_data_file' | test data file | top10_test.json |

| --word_length | word_lenght to cut and pad poem extracts | 101 |

| --max_vocab_size | max number of words in vocab  | 20_000 |

| --train_valid_ratio | ratio train valid  | 0.9 |

| --seed | Seed for replicability | 1234 |

Modeling

| Parameter  | Description | Default |
| ------------- | ------------- | ------------- |

| --epochs | number of epochs to train  | 20 |
| --seed | random seed | 1 |
| --dropout | dropout parameter | 0.5 |
| --lr | learning rate | 4e-3 | 
| --margin | triplet loss margin  | 0.24 |
| --n_classes | number of classes | 10 |
| --n_samples | number of samples | 10 |
| --data_dir | Path to data | ../data/processed/ | 
| --model_dir |  Dir for weights and model settings | ./trained_models/ | 
| --data_file | data file name | top_10_authors.json |

## Acknowledgements

The autoencoder is modified based on git repo of [blackecho](https://gist.github.com/blackecho/3a6e4d512d3aa8aa6cf9) and [Leavingseason](https://github.com/Leavingseason/rnn_recsys)  
Triplet online mining is referred from this [github](https://github.com/omoindrot/tensorflow-triplet-loss)
