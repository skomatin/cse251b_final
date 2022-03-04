################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import csv, os, sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import pandas as pd

from utils import load_vocab

class SQUAD(Dataset):

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        passage = self.df.iloc[idx, :-1][0]
        question = self.df.iloc[idx, :-1][1]
        answer = self.df.iloc[idx, -1][2]
        sample = {'passage_answer': (passage, answer), 'output': question}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Builds your datasets here based on the configuration.
# You are not required to modify this code but you are allowed to.
def get_datasets(config_data):
    train_file_path = os.path.join(sys.path[0], config_data['dataset']['training_file_path'])
    val_file_path = os.path.join(sys.path[0], config_data['dataset']['validation_file_path'])
    test_file_path = os.path.join(sys.path[0], config_data['dataset']['test_file_path'])

    # vocab_threshold = config_data['dataset']['vocabulary_threshold']
    vocabulary = load_vocab()

    train_dataset = SQUAD(csv_file=train_file_path)
    train_data_loader = DataLoader(dataset=train_dataset,
                                    batch_size=config_data['dataset']['batch_size'],
                                    shuffle=True,
                                    num_workers=config_data['dataset']['num_workers'],
                                    pin_memory=True)

    val_dataset = SQUAD(csv_file=val_file_path)
    val_data_loader = DataLoader(dataset=val_dataset,
                                    batch_size=config_data['dataset']['batch_size'],
                                    shuffle=True,
                                    num_workers=config_data['dataset']['num_workers'],
                                    pin_memory=True)

    test_dataset = SQUAD(csv_file=test_file_path)
    test_data_loader = DataLoader(dataset=test_dataset,
                                    batch_size=config_data['dataset']['batch_size'],
                                    shuffle=True,
                                    num_workers=config_data['dataset']['num_workers'],
                                    pin_memory=True)

    return vocabulary, train_data_loader, val_data_loader, test_data_loader