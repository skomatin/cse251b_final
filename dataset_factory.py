################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import csv, os, sys, pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import pandas as pd
from utils import load_vocab

import torchtext as text
from torchtext.data.utils import get_tokenizer
import constants

class SQUAD(Dataset):

    def __init__(self, file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        with open(file, 'rb') as pickle_file:
            self.data = pickle.load(pickle_file)
        self.tokenizer = get_tokenizer("basic_english")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]


# Builds your datasets here based on the configuration.
# You are not required to modify this code but you are allowed to.
def get_datasets(config_data):
    train_file_path = os.path.join(sys.path[0], config_data['dataset']['training_file_path'])
    val_file_path = os.path.join(sys.path[0], config_data['dataset']['validation_file_path'])
    test_file_path = os.path.join(sys.path[0], config_data['dataset']['test_file_path'])

    vocabulary = load_vocab()

    train_dataset = SQUAD(train_file_path)
    train_data_loader = DataLoader(dataset=train_dataset,
                                    batch_size=config_data['dataset']['batch_size'],
                                    shuffle=True,
                                    num_workers=config_data['dataset']['num_workers'],
                                    pin_memory=True)

    val_dataset = SQUAD(val_file_path)
    val_data_loader = DataLoader(dataset=val_dataset,
                                    batch_size=config_data['dataset']['batch_size'],
                                    shuffle=False,
                                    num_workers=config_data['dataset']['num_workers'],
                                    pin_memory=True)

    test_dataset = SQUAD(test_file_path)
    test_data_loader = DataLoader(dataset=test_dataset,
                                    batch_size=config_data['dataset']['batch_size'],
                                    shuffle=False,
                                    num_workers=config_data['dataset']['num_workers'],
                                    pin_memory=True)
    
    print("Loaded Dataset")
    
    return vocabulary, train_data_loader, val_data_loader, test_data_loader