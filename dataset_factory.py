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

import torchtext as text
from torchtext.data.utils import get_tokenizer
import constants

class SQUAD(Dataset):

    def __init__(self, csv_file, vocabulary):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.df = pd.read_csv(csv_file)
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = vocabulary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        passage = self.tokenizer(self.df.iloc[idx]['passage'])
        answer = self.tokenizer(self.df.iloc[idx,]['answer'])
        question = self.tokenizer(self.df.iloc[idx]['question'])

        pass_tokens = ['<start>'] + passage + ['<end>']
        ans_tokens = ['start'] + answer + ['<end>']
        q_tokens = ['<start>'] + question + ['<end>']

        pass_len = constants.MAX_PASSAGE_LEN + 2 # +2 for start and end tokens
        ans_len = constants.MAX_ANSWER_LEN + 2
        q_len = constants.MAX_QUESTION_LEN + 2

        passage = [self.vocab(word) for word in pass_tokens]
        answer = [self.vocab(word) for word in ans_tokens]
        question = [self.vocab(word) for word in q_tokens]

        # padding to same length
        pass_idxs = torch.zeros(pass_len)
        ans_idxs = torch.zeros(ans_len)
        q_idxs = torch.zeros(q_len)

        pass_idxs[:len(passage)] = torch.FloatTensor(passage)
        ans_idxs[:len(answer)] = torch.FloatTensor(answer)
        q_idxs[:len(question)] = torch.FloatTensor(question)
        
        return pass_idxs, ans_idxs, q_idxs


# Builds your datasets here based on the configuration.
# You are not required to modify this code but you are allowed to.
def get_datasets(config_data):
    train_file_path = os.path.join(sys.path[0], config_data['dataset']['training_file_path'])
    val_file_path = os.path.join(sys.path[0], config_data['dataset']['validation_file_path'])
    test_file_path = os.path.join(sys.path[0], config_data['dataset']['test_file_path'])

    vocabulary = load_vocab()

    train_dataset = SQUAD(train_file_path, vocabulary)
    train_data_loader = DataLoader(dataset=train_dataset,
                                    batch_size=config_data['dataset']['batch_size'],
                                    shuffle=True,
                                    num_workers=config_data['dataset']['num_workers'],
                                    pin_memory=True)

    val_dataset = SQUAD(val_file_path, vocabulary)
    val_data_loader = DataLoader(dataset=val_dataset,
                                    batch_size=config_data['dataset']['batch_size'],
                                    shuffle=False,
                                    num_workers=config_data['dataset']['num_workers'],
                                    pin_memory=True)

    test_dataset = SQUAD(test_file_path, vocabulary)
    test_data_loader = DataLoader(dataset=test_dataset,
                                    batch_size=config_data['dataset']['batch_size'],
                                    shuffle=False,
                                    num_workers=config_data['dataset']['num_workers'],
                                    pin_memory=True)

    return vocabulary, train_data_loader, val_data_loader, test_data_loader