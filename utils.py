import pandas as pd
import pickle
import string
import os, sys
from vocab import Vocabulary
import torch
import torchtext
from torchtext.data import get_tokenizer
import tqdm

def clean_text(text):
    """Preprocesses a text string to increase glove embedding count"""
    
    dic = {
            '$': ' $ ',
            '-': ' - ',
            '£': ' £ ',
            '₹': ' ₹ ',
            '“': ' “ ',
            '\'': ' \' ',
            '/' : ' / ',
            '[': ' [ ',
            ']': ' ] ',
            '—': ' - ',
            '–': ' - ',
            '¢': ' ¢ ',
            '‘‘': ' ‘‘ ',
            '€': ' € ',
            '<': ' < ',
            '”': ' ” ',
            '`' : ' ` ',
            '£': ' £ ',
            '+': ' + ',
            '’': ' ’ ',
            '°': ' ° ',
            '″': ' ″ ',
            '−': ' − ',
            '×': ' × '
          }
    for elem in dic.keys():
        text = text.replace(elem, dic[elem])
    return text


def save_vocab(vocab):
    with open('savedVocab', 'wb') as savedVocab:
        pickle.dump(vocab, savedVocab)
        print("Saved the vocab.")
    
def load_vocab():
    if not os.path.exists('savedVocab'):
        build_vocab()
    with open(os.path.join(sys.path[0], 'savedVocab'), 'rb') as savedVocab:
        vocab = pickle.load(savedVocab)
        print("loaded vocab")
    return vocab

def build_vocab():

    train = pd.read_csv(os.path.join(sys.path[0], './data/train.csv'))
    val = pd.read_csv(os.path.join(sys.path[0], './data/val.csv'))

    vocab = Vocabulary()
    tokenizer = get_tokenizer("basic_english")

    words = []
    print("Building Vocabulary")
    for i in range(len(train)):
        row = train.iloc[i]
        passage = str(row['passage'])
        question = str(row['question'])
        answer = str(row['answer'])
        
        passage = tokenizer(passage)
        question = tokenizer(question)
        answer = tokenizer(answer)
            
        words += passage + question + answer
    
    for i in range(len(val)):
        row = val.iloc[i]
        passage = str(row['passage'])
        question = str(row['question'])
        answer = str(row['answer'])
        
        passage = tokenizer(passage)
        question = tokenizer(question)
        answer = tokenizer(answer)
            
        words += passage + question + answer

    ser = pd.Series(words)
    counts = ser.value_counts()
    all_words = list(ser[ser.isin(counts[counts >= 2].index)].unique())
    for elem in all_words:
        vocab.add_word(elem)

    save_vocab(vocab)

    return vocab