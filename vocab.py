import string
import os, sys
import pickle
import pandas as pd
import nltk
# from collections import Counter
import functools
import operator
from constants import PAD_IDX, START_TKN

class Vocabulary(object):
    """Basic Vocabulary"""

    def __init__(self):
        self.word2idx = {'<pad>': PAD_IDX, '<start>': START_TKN, '<end>': 2, '<unk>': 3}
        self.idx2word = {PAD_IDX: '<pad>', START_TKN: '<start>', 2: '<end>', 3: '<unk>'}
        self.idx = 4

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        
    def __call__(self, word):
        if not word.lower() in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word.lower()]

    def __len__(self):
        return len(self.word2idx)
