#!/usr/bin/env python3

import torch
import torchtext
from torchtext import data
import pickle
import spacy
import random
import argparse
import os

#************************************************************
#                   About this scrip                        #
#Tokenizes, pad and slices poem extracts                    #
#Creates and stores, to train, pretained embedding          #
#************************************************************


#************************************************************
#                   Torchtext Input vars                    #
#************************************************************

parser = argparse.ArgumentParser()

def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

parser.add_argument('--data_dir', type=dir_path, default = '../data/processed/',
                        help="dir of data (default: ../data/wrangled/)")

parser.add_argument('--train_data_file', type=str, default = 'top10_train.json',
                        help="train data file (default: top10_train_.json)")

parser.add_argument('--test_data_file', type=str, default = 'top10_test.json',
                        help="test data file (default: top10_test_.json)")

parser.add_argument('--word_length', type=int, default=101,
                        help='word_lenght to cut and pad poem extracts (default: 101)')

parser.add_argument('--max_vocab_size', type=int, default = 20000,
                        help="max number of words in vocab  (default: 20_000)")

parser.add_argument('--train_valid_ratio', type=float, default = 0.9,
                        help="ratio train valid  (default: 0.9)")

parser.add_argument('--seed', type=int, default = 1234,
                        help="Seed for replicability (default: 1234)")


args = parser.parse_args()
#************************************************************


#************************************************************
#                   Processing Input vars                   #
#                   All tokenization options                #
#************************************************************

data_dir = args.data_dir
train_data_file = args.train_data_file
test_data_file = args.test_data_file
WORD_LENGTH = args.word_length
MAX_VOCAB_SIZE = args.max_vocab_size
SEED=args.seed
TRAIN_VALID_RATIO =args.train_valid_ratio


#************************************************************


assert  train_data_file[-11:]== '_train.json'

prefix=train_data_file[:-11]

file_embedding = prefix + '_embedding.pkl'
file_dict = prefix + '_dict.pkl'
tokenized_train_data = prefix + '_train.pkl'
tokenized_valid_data = prefix + '_valid.pkl'
tokenized_test_data = prefix + '_test.pkl'
pretrained_embedding = 'glove.6B.100d'
TOKENIZE = 'spacy'


# Create data fieds for poems extracts and labels
TEXT = data.Field(lower=True,include_lengths=False,
tokenize = TOKENIZE)
LABEL = data.Field( dtype = torch.int)
dataFields = {'content': ('content', TEXT),
              'author_label': ('author_label', LABEL)}

#Tokenize data
path_train_data= os.path.join(data_dir,train_data_file)
train_dataset= data.TabularDataset(path=path_train_data,
                                            format='json',
                                            fields=dataFields)
path_test_data= os.path.join(data_dir,test_data_file)
test_data= data.TabularDataset(path=path_test_data,
                                            format='json',
                                            fields=dataFields)
#split train into train and valid
train_data, valid_data = train_dataset.split(split_ratio=TRAIN_VALID_RATIO,
random_state = random.seed(SEED))

#Build train set vocab
TEXT.build_vocab(train_dataset,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = pretrained_embedding,
                 unk_init = torch.Tensor.normal_)

#token dictionary
word_dict = dict(TEXT.vocab.stoi)
#dump dictionary
path_to_dict = os.path.join(data_dir+file_dict)
with open(path_to_dict, 'wb') as f:
    pickle.dump(word_dict, f)

#dump embedding
path_to_embedding = os.path.join(data_dir,file_embedding)
with open(path_to_embedding, 'wb') as f:
    pickle.dump(TEXT.vocab.vectors.tolist(), f)

#Process word extracts to be compused of n words,
#path and slice accordingling
def slice_and_pad_to_n(ls, n):
    """ls list to be cut/padded to length n"""
    ls= ls[:n]
    if len(ls)<n: ls=(['<pad>']*(n-len(ls))) + ls
    return ls

def tokenize_poem(token_list, word_dict):
    """
    List to tockens
    word_dict: dictionary of tokens
    token_list: word list
    """
    aux =[word_dict[w] if w in word_dict else word_dict['<unk>'] for w in token_list]
    return aux

def tokenize_pad_slice_poems(data, word_length, word_dict):
    """
    data: poem extract as list
    word_length: final word length of poems
    token_dict: tokens dictionary
    tokenized_poems: tokenized poems
    """
    data = [slice_and_pad_to_n(p, word_length) for p in data]
    tokenized_poems = [tokenize_poem(p ,word_dict) for p in data]
    return tokenized_poems

def process_and_save(data, word_length, word_dict, data_dir, file_name):
    """
    data : poems list created by spacy and torchtext
    word_length: fix poems to have this number of tokens (not only words)
    data_dir: dir to store data_dir
    file_name: file name to store data
    """
    data_list = [i.content for i in data.examples]
    labels_list = [i.author_label for i in data.examples]
    poems_list = tokenize_pad_slice_poems(data_list, word_length, word_dict)
    path_to_file = os.path.join(data_dir, file_name)
    with open(path_to_file, 'wb') as f:
        pickle.dump((poems_list,labels_list), f)

process_and_save(train_data, WORD_LENGTH, word_dict, data_dir, tokenized_train_data)
process_and_save(valid_data, WORD_LENGTH, word_dict, data_dir, tokenized_valid_data)
process_and_save(test_data, WORD_LENGTH, word_dict, data_dir, tokenized_test_data)
