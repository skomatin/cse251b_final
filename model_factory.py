import torch
import torch.nn as nn
import torchvision
from bidaf_lstm import *
from v_transformer import *
from baseline import *
import constants

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    
    # You may add more parameters if you want
    num_layers = config_data['model']['num_layers']
    model_temp = config_data['generation']['temperature']
    question_length = constants.MAX_QUESTION_LEN + 2
        
    # Define and return model
    if model_type == 'BiDAFLSTM':
        return BiDAF_LSTMNet(embedding_size, hidden_size, num_layers, vocab, question_length, model_temp)
    elif model_type == 'v_transformer':
        return VTransformer(
            config_data['transformer']['num_encoder_layers'],
            config_data['transformer']['num_decoder_layers'],
            embedding_size,
            config_data['transformer']['nhead'],
            vocab,
            config_data['transformer']['dim_feedforward'],
            config_data['transformer']['dropout']
    )
    elif model_type == 'baseline':
        return base_LSTM(
            hidden_size,
            embedding_size,
            num_layers,
            vocab,
            model_temp
        )
