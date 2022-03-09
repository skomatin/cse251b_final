import torch
import numpy as np
import torch.nn as nn
import math
from constants import *
from torch.cuda.amp import autocast
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # https://pytorch.org/tutorials/beginner/translation_transformer.html
    """
    PositionalEncoding module injects some information about the relative or absolute position 
    of the tokens in the sequence. The positional encodings have the same dimension as the 
    embeddings so that the two can be summed. Here, we use sine and cosine functions of 
    different frequencies.
    """

    def __init__(self, embedding_dim, dropout, max_len=20):
        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)        # max_len x 1 x embedding_dim     

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pe)

    def forward(self, token_embedding):
        # token_embedding: N x max_len x embedding_dim
        embeds = token_embedding.transpose(0,1) + self.pos_embedding[:(token_embedding.transpose(0,1)).size(0), :]
        embeds = embeds.transpose(0,1)      # N x max_len x embedding_dim
        return self.dropout(embeds)

    def __call__(self, token_embedding):
        return self.forward(token_embedding)

class TokenEmbedding(nn.Module):
    # https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)    # N x L x E 

    def forward(self, x):
        """
        The reason we increase the embedding values before addition is to make the 
        positional encoding relatively smaller. This means the original meaning 
        in the embedding vector won't be lost when we add them together.
        """

        # Input : N x L 
        return self.embedding(x.long()) * math.sqrt(self.embedding_dim)

    def __call__(self, x):
        return self.forward(x)

class VTransformer(nn.Module):

    def __init__(self, num_encoder_layers, num_decoder_layers, embedding_dim, nhead, \
                    vocab, dim_ff, dropout, model_temp=0.1):
        """
        nhead = number of heads in the multihead attention model
        dim_ff = dimension of the feed forward layer
        """
        super().__init__()

        self.model_temp = model_temp
        self.transformer = nn.Transformer(
            d_model = embedding_dim,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = dim_ff,
            dropout = dropout,
            activation = 'relu',
            batch_first = True,
            norm_first = False
        )
        # TODO: pass max_len as param from the model init file
        vocab_size = len(vocab)
        max_len = (MAX_PASSAGE_LEN+2)+(MAX_ANSWER_LEN+2)+(MAX_QUESTION_LEN+2)
        self.generator = nn.Linear(embedding_dim, vocab_size)
        self.embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout, max_len)

    
    def embed(self, tokens):
        embedded_tokens = self.positional_encoding(self.embedding(tokens))      # N x L x E
        return embedded_tokens
    
    # @autocast()
    def forward(self, passage, answer, question):

        # The model needs to be given tokens from <sos> to token before <eos> in order to be able to predict the next token
        question_input = question[:, :-1]
        # The model is expected to output tokens starting from the second token given first token as input
        question_expected = question[:, 1:]

        passage_ans_q = torch.cat([passage, answer, question_input], dim=1)       # N x max_len = 64 x 963-1
        pass_ans_dim = passage.shape[1] + answer.shape[1]       # = 901
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(passage_ans_q[:, :pass_ans_dim], question_input)

        passage_ans_q_emd = self.embed(passage_ans_q)   # N x L x E

        # output = [N x passage_ans_dim x embedding_dim], [N x max_question_len-1 x embedding_dim]
        passage_ans_emd, question_emd = passage_ans_q_emd[:,:pass_ans_dim,:], passage_ans_q_emd[:,pass_ans_dim:,:]

        # output = [N x max_question_len-1 x embedding_dim]
        out = self.transformer(src=passage_ans_emd, tgt=question_emd, src_mask=src_mask, \
                    tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, \
                    tgt_key_padding_mask=tgt_padding_mask)
        
        # output = [N x max_question_len-1 x vocab_len]
        # no softmax since loss will compute it
        out = self.generator(out)

        ######################### TODO!!!!!!
        # Can I add an <eos> token after this to equate the length to questions? If done, remove the second arg
        # out = torch.cat([torch.tensor([[[START_TKN]]], dtype=torch.long, device=device).transpose(1,2), out], dim=1)
        # add_start = torch.tensor([START_TKN], dtype=torch.long, device=device)
        # add_start = (add_start.unsqueeze(0)).unsqueeze(2)
        # out = torch.cat([add_start, out], dim=1)
        # print (f"Sample out value: {out[0]}")

        return out, question_expected


    def encode(self, src, src_mask):
        pos_emb = self.positional_encoding(self.embedding(src))
        return self.transformer.encoder(pos_emb, src_mask)

    def decode(self, tgt, memory, tgt_mask):
        pos_emb = self.positional_encoding(self.embedding(tgt))
        return self.transformer.decoder(pos_emb, memory, tgt_mask)

    def predict(self, passage, answer):
        with torch.no_grad():
            bsz = passage.shape[0]
            question_input = torch.tensor([[START_TKN]], dtype=torch.long, device=device)
            # question_input = torch.ones(1, 1).fill_(START_TKN).type(torch.long).to(device)

            for _ in range(MAX_QUESTION_LEN+1):
                passage_ans = torch.cat([passage, answer], dim=1)   # N x 901

                # Embedding
                passage_ans_q = torch.cat([passage, answer, question_input], dim=1)
                pass_ans_dim = passage.shape[1] + answer.shape[1]

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(passage_ans_q[:, :pass_ans_dim], question_input)

                passage_ans_q_emd = self.embed(passage_ans_q)   # N x L x E

                # output = [N x passage_ans_dim x embedding_dim], [N x max_question_len-1 x embedding_dim]
                passage_ans_emd, question_emd = passage_ans_q_emd[:,:pass_ans_dim,:], passage_ans_q_emd[:,pass_ans_dim:,:]

                # Forward
                out = self.transformer(src=passage_ans_emd, tgt=question_emd, src_mask=src_mask, \
                    tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, \
                    tgt_key_padding_mask=tgt_padding_mask)
                out = self.generator(out[:, -1])    # 1 x vocab_size

                # Next predicted work is the one with the highest probability.. Softmax? TODO
                # next_token = out.topk(1)[1].view(-1)[-1].item()
                # next_token = torch.max(out, dim=1)
                probs = nn.Softmax(dim=1)(out.div(self.model_temp)).squeeze() # N x vocab_size
                next_token = torch.multinomial(probs, 1).view(-1,1)  # N x 1
                # next_token = torch.tensor([[next_token]], device=device)

                question_input = torch.cat([question_input, next_token], dim=1)
                # question_input = torch.cat([question_input, torch.ones(1, 1).type_as(passage.data).fill_(next_token)], dim=1)

                # If the model predicts <eos>, break out of prediction loop
                if next_token == END_TKN:
                    break
                
            # return torch.tensor(question_input.view(-1))
            return question_input

    def __call__(self, passage, answer, question):
        return self.forward(passage, answer, question)



def generate_square_subsequent_mask(sz):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)   # T x T
    # mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1)       
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) # T x T
    return mask


def create_mask(src, tgt=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # src: N x max_possible_len_of_source(S) = N x 901
    src_seq_len = src.shape[1]
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)   # S x S
    src_padding_mask = (src == PAD_IDX)     # N x S
    
    tgt_mask = None
    tgt_padding_mask = None

    if tgt is not None:
        # tgt: N x T = N x 62
        tgt_seq_len = tgt.shape[1]  # T
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)     # T x T
        tgt_padding_mask = (tgt == PAD_IDX)                         # N x T

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
