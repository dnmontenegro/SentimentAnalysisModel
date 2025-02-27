'''
This program defines a class that reads data from a csv file and loads input features and labels for a given index.
'''
from torch.utils.data import Dataset
import pandas
from ast import literal_eval
import torch

class ReviewDataset(Dataset):
    def __init__(self, filename):
        self.df = pandas.read_csv(filename, converters={'input_x': literal_eval})
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        input_x = self.df.loc[index, 'input_x'] # Load input features
        label = self.df.loc[index, 'Label'] # Load labels
        return torch.tensor(input_x), torch.tensor(label, dtype=torch.float)
        
        