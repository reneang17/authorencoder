import json
import os
import torch

def _get_data(data_prefix, data_dir='./data_to_s3', with_labels= True):
    #print("Get train data loader.")
    with open(os.path.join(data_dir,
              data_prefix+'_text_list.json'), 'r') as f:
        train_X = json.load(f)
    train_X = [torch.tensor(t) for t in train_X] #.long()
    
    if with_labels:
        with open(os.path.join(data_dir,
                  data_prefix+'_labels_list.json'), 'r') as f:
            train_y = json.load(f)

        train_y = [torch.tensor(t) for t in train_y] # .float().squeeze()
        return train_X , train_y
    
    else: 
        return train_X
    
class Data_iterator:
    def __init__(self, data_prefix, data_dir='./data_to_s3'):
        self.X, self.y  = _get_data(data_prefix, data_dir)
        
    def __iter__(self):
        return iter(zip(self.X, self.y))
    
class Test_iterator:
    def __init__(self, data_prefix, data_dir='./data_to_s3'):
        self.X = _get_data(data_prefix, data_dir, with_labels= False)
        
    def __iter__(self):
        return iter(self.X)