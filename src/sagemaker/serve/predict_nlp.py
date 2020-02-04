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



from utils_nlp import tokenize

from model_nlp import CNN

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
    #model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], 
    #model_info['vocab_size'])
    
    INPUT_DIM =  20002 # len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [2,3,4]
    OUTPUT_DIM = 6
    DROPOUT = 0.5
    PAD_IDX = 1 # TEXT.vocab.stoi[TEXT.pad_token]
    
    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, 
                FILTER_SIZES, OUTPUT_DIM, DROPOUT, 
                PAD_IDX)
    
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

def translate_labels(labels_list):
    toxic_labels = ["toxic", "severe_toxic", 
                    "obscene", "threat", "insult", 
                    "identity_hate"]
    output_string = 'Your text has been classified as:\n'
        
    for j,i in enumerate(labels_list):
        if i==1.: output_string+= toxic_labels[j]+'\n'
        else: output_string+= 'Not '+toxic_labels[j]+'\n'
    return output_string 

def predict_fn(input_text, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    word_dict = model.word_dict
    
    
    tokenized = tokenize(word_dict, input_text)

    tensor = torch.LongTensor(tokenized).to(device)
    tensor = tensor.unsqueeze(1)
    
    
    # Make sure to put the model into evaluation mode
    model.eval()

    #raise Exception('This is the input: ' + tensor)

    with torch.no_grad():
        output = model.forward(tensor)
        prediction = torch.sigmoid(output).squeeze().detach()

    result = np.round(prediction).tolist()

    return translate_labels(result)
