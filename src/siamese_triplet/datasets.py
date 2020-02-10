import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

import pickle
import os

np.random.seed(1234)

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
    
    
def data_to_Iterator(data_dir, file_name, n_classes=10, n_samples=10, sampler= True):

    with open(os.path.join(data_dir, file_name) , 'rb') as f:
        data, labels = pickle.load(f)
        
    data = [torch.LongTensor(i) for i in data]
    data_tensor = torch.stack(data)
    labels_tensor= torch.Tensor(labels)
    tensordataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)
    
    if sampler:
        Batch_sampler = BalancedBatchSampler(labels_tensor, n_classes=10, n_samples=10)
        Loader=torch.utils.data.DataLoader(tensordataset, sampler=Batch_sampler, pin_memory=True)
    else:
        Loader=torch.utils.data.DataLoader(tensordataset, pin_memory=True)
    return Loader 
    

