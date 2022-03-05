import torch
import torch.nn as nn
import torchvision

from models import *


# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    
    # You may add more parameters if you want
    num_layers = config_data['model']['num_layers']
    vocab_size = len(vocab)
    model_temp = config_data['generation']['temperature']
        
    # Define and return model
    if model_type == 'LSTM':
        return EncoderDecoderLSTM(hidden_size, embedding_size, num_layers, vocab_size, model_temp)
    else:
        return None

