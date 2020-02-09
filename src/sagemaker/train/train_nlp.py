import argparse
import json
import os
import pickle
import sys
import sagemaker_containers

import torch.nn as nn
import numpy as np
import torch

from models import CNN

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




if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    # Training Parameters
    #parser.add_argument('--batch-size', type=int, default=512, metavar='N',
    #                    help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 1)')    
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='S',
                        help='dropout parameter')
    

    # Model Parameters
    #parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
    #                    help='size of the word embeddings (default: 32)')
    #parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
    #                    help='size of the hidden dimension (default: 100)')
    #parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
    #                    help='size of the vocabulary (default: 5000)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    import numpy as np 
    import torch
    from torch.optim import lr_scheduler
    import torch.optim as optim
    from torch.autograd import Variable
    
    from siamese_triplet import OnlineTripletLoss
    from siamese_triplet import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector
    from siamese_triplet import AverageNonzeroTripletsMetric
    from siamese_triplet import simplified_fit
    from siamese_triplet import BalancedBatchSampler, data_to_Iterator
    
    import sys
    import pickle
    sys.path.insert(1, './')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Using device {}.".format(device))

    

    
    torch.manual_seed(args.seed)

    #data, embedding, word_dict files names.
    
    data_name = 'longest10.json'
    prefix = data_name[:-5]

    file_embedding = prefix + '_embedding.pkl'
    file_dict = prefix + '_dict.pkl'
    file_tokenized_train_data = prefix + '_train.pkl'
    file_tokenized_valid_data = prefix + '_valid.pkl'
    
     

    # Build the model ***********************************************
    
    #Load dictionary
    with open(os.path.join(args.data_dir, file_dict), "rb") as f:
        word_dict = pickle.load(f)
    
    with open(os.path.join(args.data_dir, file_embedding), 'rb') as f:
        vocab_vectors = pickle.load(f)
    
    
    
    vocab_vectors = torch.tensor(vocab_vectors)
    
    INPUT_DIM = len(word_dict)
    WORD_EMBEDDING_DIM = vocab_vectors.size(1)  #Fixed by preloaded embedding 
    N_FILTERS = 100
    FILTER_SIZES = [2,3,4]
    AUTHOR_DIM = 2
    DROPOUT = args.dropout
    PAD_IDX = 1
    #UNK_IDX = 0
    print(WORD_EMBEDDING_DIM)
    model = CNN(INPUT_DIM, WORD_EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, AUTHOR_DIM, DROPOUT, PAD_IDX)

    #Load dictionary into model
    model.word_dict = word_dict
    
    #Load dictionary into model
    model.embedding.weight.data.copy_(vocab_vectors)

    print("Model loaded with WORD_EMBEDDING_DIM {}, N_FILTERS {}, vocab_size {}.".format(
    #    args.embedding_dim, args.hidden_dim, args.vocab_size
         WORD_EMBEDDING_DIM, N_FILTERS, INPUT_DIM))
    
    # Build the model ends *****************************************
    
    # Train the model.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
       
    
    # Load and prepare data ***********************
    
    train_Loader=data_to_Iterator(args.data_dir, file_tokenized_train_data, n_classes=10, n_samples=10, sampler= True)
    valid_Loader=data_to_Iterator(args.data_dir, file_tokenized_valid_data, n_classes=10, n_samples=10, sampler= True)
    print('Train/Valid Data Loaded')
    #*********************
    
    MARGIN = 0.25
    MODEL= model.to(device)
    loss_fn = OnlineTripletLoss(MARGIN, RandomNegativeTripletSelector(MARGIN))
    LR = 5e-3 
    optimizer = optim.Adam(MODEL.parameters(), lr=LR, weight_decay=1e-4)
    n_epochs = 15
    log_interval = 4

    N_EPOCHS = args.epochs

    # training
    is_cuda_available= False
    simplified_fit(train_Loader, valid_Loader, MODEL, loss_fn, optimizer, N_EPOCHS, \
                   is_cuda_available, metrics=[AverageNonzeroTripletsMetric()])
        

    #Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'INPUT_DIM' : INPUT_DIM,
            'WORD_EMBEDDING_DIM' : WORD_EMBEDDING_DIM, #Fixed by preloaded embedding 
            'N_FILTERS' : N_FILTERS,
            'FILTER_SIZES' : FILTER_SIZES,
            'AUTHOR_DIM' : AUTHOR_DIM,
            'DROPOUT' : DROPOUT,
            'PAD_IDX' : PAD_IDX
        }
        print("model_info: {}".format(model_info))
        torch.save(model_info, f)

    
    with open(os.path.join(args.data_dir, file_tokenized_train_data) , 'rb') as f:
        data_set = pickle.load(f)
        
    training_set_path = os.path.join(args.model_dir, 'training_set.pkl')
    with open(training_set_path, 'wb') as f:
        pickle.dump(data_set, f)
    
    #Save the trained vocab embedding
    word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model_state.pt')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
