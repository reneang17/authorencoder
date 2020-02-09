import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy



from utils_nlp import tokenize, emb

from models import CNN
from sklearn.neighbors import KNeighborsClassifier as KNC


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #**********************************    
    # Load model
    
    INPUT_DIM = model_info['INPUT_DIM']
    WORD_EMBEDDING_DIM = model_info['WORD_EMBEDDING_DIM']  
    N_FILTERS = model_info['N_FILTERS']
    FILTER_SIZES = model_info['FILTER_SIZES']
    AUTHOR_DIM = model_info['AUTHOR_DIM']
    DROPOUT = model_info['DROPOUT']
    PAD_IDX = model_info['PAD_IDX']
    #UNK_IDX = 0
    
    
    model = CNN(INPUT_DIM, WORD_EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, AUTHOR_DIM, DROPOUT, PAD_IDX)
    

    print("Model loaded with embedding_dim {}, vocab_size {}.".format(
    #    args.embedding_dim, args.hidden_dim, args.vocab_size
         WORD_EMBEDDING_DIM, INPUT_DIM))
    
    #**********************************
    
    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model_state.pt')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)
           
    model.to(device).eval()

    print("Done loading model.")
    return model





def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

#import training data and form clusters


#author_encoder_path = os.path.join('./', 'authorencoder.pkl')
#with open(author_encoder_path, 'rb') as f:
#    train_embeddings_otl, train_labels_otl= pickle.load( f ) 

train_embeddings_otl, train_labels_otl = emb()


from sklearn.neighbors import KNeighborsClassifier as KNC
KNN = KNC(n_neighbors=3)
KNN.fit(train_embeddings_otl, train_labels_otl)


def predict_fn(input_text, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
        
    model = model
    
    word_dict = model.word_dict
    
    tokenized = tokenize(word_dict, input_text)

    tensor = torch.tensor(tokenized).to(device)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    
    # Make sure to put the model into evaluation mode
    model.eval()

    #raise Exception('This is the input: ' + tensor)

    with torch.no_grad():
        output = model.forward(tensor).tolist()
    prediction = int(KNN.predict(output).item())

    author_dir = {0: 'John Dryden', 1: 'Robert Pinsky', 2: 'Anne Carson', 3: 'Alfred Lord Tennyson', 4: 'Allen Ginsberg', 5: 'Philip Whalen', 6: 'Matthew Arnold', 7: 'Walt Whitman', 8: 'William Shakespeare', 9: 'Beowulf Anonimous'}

    return author_dir[prediction]
