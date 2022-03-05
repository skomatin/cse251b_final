import string
import os, sys
import pickle
import pandas as pd
import nltk
# from collections import Counter
import functools
import operator

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


# def load_vocab(threshold, config_data):
#     vocab_file = os.path.join(sys.path[0], 'savedVocab')
#     if os.path.isfile(vocab_file):
#         with open(vocab_file, 'rb') as savedVocab:
#             vocab = pickle.load(savedVocab)
#             print("Using the saved vocab.")

#     else:
#         vocab = build_vocab(threshold, config_data)
#         with open(vocab_file, 'wb') as savedVocab:
#             pickle.dump(vocab, savedVocab)
#             print("Saved the vocab.")

#     return vocab

# #TODO how are we dealing with question vocab
# def build_vocab(threshold, config_data):
#     print("Building Vocab")
#     train_file_path = os.path.join(sys.path[0], config_data['dataset']['training_file_path'])

#     train_df = pd.read_csv(train_file_path)
#     input_list = train_df.iloc[:, 0].tolist() + train_df.iloc[:, 2].tolist()
#     input_list = list(map(nltk.tokenize.word_tokenize, input_list))
#     input_list = functools.reduce(operator.iconcat, input_list, [])
#     input_list = list(map(str.lower, input_list))

#     ser = pd.Series(input_list)
#     counts = ser.value_counts()
#     words = list(ser[ser.isin(counts[counts >= threshold].index)].unique())

#     # Create a vocab wrapper and add some special tokens.
#     #TODO are these still needed?
#     vocab = Vocabulary()
#     vocab.add_word('<pad>')
#     vocab.add_word('<start>')
#     vocab.add_word('<end>')
#     vocab.add_word('<unk>')

#     # Add the words to the vocabulary.
#     for i, word in enumerate(words):
#         vocab.add_word(word)
#     return vocab