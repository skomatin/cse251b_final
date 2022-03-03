import string
import os
import pickle

class Vocabulary(object):
    """Basic Vocabulary"""

    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {}
        self.idx = 2

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

    def save(self):
        with open('savedVocab', 'wb') as savedVocab:
            pickle.dump(self, savedVocab)
            print("Saved the vocab.")
    
def load_vocab(self):
    with open('savedVocab', 'rb') as savedVocab:
        vocab = pickle.load(savedVocab)
        print("loaded vocab")
    return vocab