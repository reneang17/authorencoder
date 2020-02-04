import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
#import pandas as pd
#import torch
#import torch.optim as optim
#import torch.utils.data
import torch.nn as nn
#from sklearn.metrics import roc_auc_score
import numpy as np
import torch



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
    #model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])
    
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




if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    #parser.add_argument('--batch-size', type=int, default=512, metavar='N',
    #                    help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    #train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    #iterator_train = Data_iterator('train', training_dir=args.data_dir )
    #iterator_val = Data_iterator('val', training_dir=args.data_dir )
    

    # Build the model.
    INPUT_DIM =  20002 # len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [2,3,4]
    OUTPUT_DIM = 6
    DROPOUT = 0.5
    PAD_IDX = 1 # TEXT.vocab.stoi[TEXT.pad_token]

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, 
            FILTER_SIZES ,OUTPUT_DIM, DROPOUT, PAD_IDX)

    with open(os.path.join(args.data_dir, "word_dict.pkl"), "rb") as f:
        model.word_dict = pickle.load(f)
    
    with open(os.path.join(args.data_dir,'untrained_vocab_vectors_list.json'), 'r') as f:
        vocab_vectors = json.load(f)
    model.embedding.weight.data.copy_(torch.tensor(vocab_vectors))

    print("Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}.".format(
    #    args.embedding_dim, args.hidden_dim, args.vocab_size
         EMBEDDING_DIM, EMBEDDING_DIM, INPUT_DIM
    ))

    
    
    # Train the model.
    import torch.optim as optim
    # optimizer = optim.Adam(model.parameters())
    # loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    
    
    def _get_data(data_prefix, training_dir, with_labels= True):
    #print("Get train data loader.")
        with open(os.path.join(training_dir, 
            data_prefix+'_text_list.json'), 'r') as f:
            train_X = json.load(f)
        train_X = [torch.tensor(t) for t in train_X] #.long()
    
        if with_labels:
            with open(os.path.join(training_dir,
                  data_prefix+'_labels_list.json'), 'r') as f:
                train_y = json.load(f)

            train_y = [torch.tensor(t) for t in train_y] # .float().squeeze()
            return train_X , train_y    
        else: 
            return train_X

    class Data_iterator:
        def __init__(self, data_prefix, training_dir):
            self.X, self.y  = _get_data(data_prefix, training_dir)
        
        def __iter__(self):
            return iter(zip(self.X, self.y))

    iterator_train = Data_iterator('train', training_dir=args.data_dir )
    iterator_val = Data_iterator('val', training_dir=args.data_dir )
    
#def roc_auc(preds, y):
#    """
#    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
#    """
#    acc = roc_auc_score(y, preds)  
#    return acc


    def train(model, iterator, optimizer, criterion, device):
    
        epoch_loss = 0
        epoch_acc = 0
    
        model.train()

 
        iterations=0
        for batch in iterator:
            iterations+=1
                
            batch_X, batch_y = batch
        
            batch_X=batch_X.to(device)
            batch_y=batch_y.to(device)
        
            optimizer.zero_grad()
            
        
            predictions = model(batch_X).squeeze(1)
        
            loss = criterion(predictions, batch_y)
        
            loss.backward()
        
            optimizer.step()
        
            epoch_loss += loss.item()
        
        
        
        return epoch_loss / iterations#, roc_auc(np.vstack(preds_list), np.vstack(labels_list))

    def evaluate(model, iterator, criterion, device):
    
        epoch_loss = 0
        epoch_acc = 0
    
        model.eval()

    
        with torch.no_grad():
            iterations = 0
            for batch in iterator:
                iterations+=1
            
                batch_X, batch_y = batch
            
                batch_X=batch_X.to(device)
                batch_y=batch_y.to(device)
            
                predictions = model(batch_X).squeeze(1)
            
                loss = criterion(predictions, batch_y)

                epoch_loss += loss.item()
        
        return epoch_loss / iterations#, roc_auc(np.vstack(preds_list), np.vstack(labels_list))
    

    #train(model, train_loader, args.epochs, optimizer, loss_fn, device)
    
    model.embedding.weight.requires_grad = True


    N_EPOCHS = args.epochs

    best_valid_loss = float('inf')

    import time

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    for epoch in range(N_EPOCHS):

        start_time = time.time()       
        train_loss = train(model, iterator_train, optimizer, criterion, device)
        valid_loss = evaluate(model, iterator_val, criterion, device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print('\tEpoch:', epoch, 'Epoch Time: ', epoch_mins, 'm ', epoch_secs, 's')
        print('\tTrain Loss: ', train_loss )
        print('\tVal. Loss:  ', valid_loss )
        
#        if valid_loss < best_valid_loss:
#            best_valid_loss = valid_loss
#            model_path = os.path.join(args.model_dir, 'model_state.pt')


    #Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            #'embedding_dim': args.embedding_dim,
            #'hidden_dim': args.hidden_dim,
            #'vocab_size': args.vocab_size,
            'vocab_size': 20002,
        }
        torch.save(model_info, f)

    #Save the trained vocab embedding
    word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model_state.pt')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)

    #Save the trained vocab embedding
    #word_dict_path = os.path.join(args.model_dir, 'trained_vocab.pkl')
    #with open(word_dict_path, 'w') as f:
    #    json.dump(model.embedding.weight.data.tolist(), f)