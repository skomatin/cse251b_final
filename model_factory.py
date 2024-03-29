import torch
import torch.nn as nn
import torchvision
from bidaf_lstm import *
from v_transformer import *
from baseline import *
from baseline_1 import *
from baseline_2 import *
import constants
# from custom import *
# from custom_masked import *
# from attentional_lstm import *

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    
    # You may add more parameters if you want
    num_layers = config_data['model']['num_layers']
    model_temp = config_data['generation']['temperature']
    question_length = constants.MAX_QUESTION_LEN + 2
        
    model = None
    # Define and return model
    if model_type == 'BiDAFLSTM':
        model = BiDAF_LSTMNet(embedding_size, hidden_size, num_layers, vocab, question_length, model_temp)
    elif model_type == 'BasicModel':
        model = BasicQuestioner(embedding_size, hidden_size, num_layers, vocab, model_temp)
    elif model_type == 'BasicModelMasked':
        model = BasicQuestionerMasked(embedding_size, hidden_size, num_layers, vocab, model_temp)
    elif model_type == 'v_transformer':
        model = VTransformer(
            config_data['transformer']['num_encoder_layers'],
            config_data['transformer']['num_decoder_layers'],
            embedding_size,
            config_data['transformer']['nhead'],
            vocab,
            config_data['transformer']['dim_feedforward'],
            config_data['transformer']['dropout']
        )
    elif model_type == 'AttentionalLSTM':
        model = AttentionalQuestioner(embedding_size, hidden_size, 
            config_data['model']['num_encoder_layers'],
            config_data['model']['num_decoder_layers'],
            config_data['model']['num_encoder_heads'],
            vocab,
            model_temp)
    elif model_type == 'baseline':
        model = base_LSTM(
            hidden_size,
            embedding_size,
            num_layers,
            vocab,
            model_temp
        )
    elif model_type == 'baseline_1':
        model = base_LSTM1(
            hidden_size,
            embedding_size,
            num_layers,
            vocab,
            model_temp
        )
    elif model_type == 'baseline_2':
        model = base_GRU(
            hidden_size,
            embedding_size,
            num_layers,
            vocab,
            model_temp
        )

    print('Loaded Model')
    return model