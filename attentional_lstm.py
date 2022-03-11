from unicodedata import bidirectional
import torch
import torch.nn as nn
import numpy as np
from constants import *

class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_encoder_layers, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads

        self.context_extractor = nn.LSTM(self.embed_dim, self.hidden_size, self.num_encoder_layers, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(2*self.hidden_size, self.num_heads, batch_first=True)
        self.sm = nn.Softmax(dim=2)

        combined_length = MAX_SHORT_ANSWER_LEN + MAX_SHORT_PASSAGE_LEN + 2
        self.word_transform = nn.Sequential(
            nn.Linear(combined_length, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, passage, answer, mask):
        """
        passage: N x P x E
        answer: N x A x E
        mask: N x A
        """
        bs, P = passage.shape[0], passage.shape[1]
        pass_embed, _ = self.context_extractor(passage) # N x P x 2H
        ans_embed, _ = self.context_extractor(answer) # N x A x 2H

        # C2Q of multi-head attention?
        # sim = torch.bmm(pass_embed, ans_embed.transpose(1, 2)) # N x P x A
        # a = self.sm(sim)
        # context = torch.bmm(a, ans_embed) # N x P x 2H
        
        context, atten_weights = self.attention(query=pass_embed, key=ans_embed, value=ans_embed, key_padding_mask=mask) # N x P x 2H switch answer and passage?
        context = context.view(bs, P, 2, self.hidden_size)[:, :, 0, :] # N x P x H
        
        return context

    def __call__(self, passage, answer, mask):
        return self.forward(passage, answer, mask)

class GlobalAttention(nn.Module):
    
    def __init__(self, embed_dim, hidden_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.sm = nn.Softmax(dim=1)
        self.fc = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()

    def mask_context(self, context_mask):
        """N x S"""
        self.context_mask = context_mask

    def forward(self, context, input):
        """
        context: N x S x H
        input: N x 1 x H

        Dot-product Attention https://arxiv.org/pdf/1508.04025v2.pdf
        """
        scores = torch.bmm(context, input.transpose(1, 2)) # N x S x 1
        scores = scores.data.masked_fill_(self.context_mask.unsqueeze(2), -float('inf')) # masks out the padding source tokens
        alphas = self.sm(scores) # N x S x 1 
        attended_context = torch.bmm(alphas.transpose(1, 2), context) # N x 1 x H
        concatenated = torch.cat([attended_context, input], dim=2) # N x 1 x 2H
        output = self.fc(concatenated) # N x 1 x H
        output = self.tanh(output)
        return output

    def __call__(self, context, input):
        return self.forward(context, input)

class Decoder(nn.Module):
    
    def __init__(self, embed_dim, hidden_size, num_decoder_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_decoder_layers = num_decoder_layers

        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, batch_first=True)

    def forward(self, input, hidden_state=None):
        """N x 1 x E"""
        dec_out, hidden_state = self.lstm(input, hidden_state)
        return dec_out, hidden_state

    def __call__(self, input, hidden_state=None):
        return self.forward(input, hidden_state)

class AttentionalQuestioner(nn.Module):

    def __init__(self, embedding_dim, hidden_size, num_encoder_layers, num_decoder_layers, num_encoder_heads, vocab, model_temp):
        super().__init__()
        self.embed_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_heads = num_encoder_heads
        self.vocab_size = len(vocab)
        self.model_temp = model_temp

        self.embed_mat = torch.from_numpy(np.load('embeddings.npy')).float()
        self.embedder = nn.Embedding.from_pretrained(self.embed_mat, freeze=True, padding_idx=0)

        self.encoder = Encoder(self.embed_dim, self.hidden_size, self.num_encoder_layers, self.num_encoder_heads)
        self.atten = GlobalAttention(self.embed_dim, self.hidden_size)
        self.decoder = Decoder(self.embed_dim, self.hidden_size, self.num_decoder_layers)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        self.sm = nn.Softmax(dim=2)

    def forward(self, passage, answer, question):
        """
        passage, answer, question: N x P, N x A, N x Q
        """
        passage_embed, answer_embed, q_emb = self.embedder(passage), self.embedder(answer), self.embedder(question)
        encoder_mask = (answer == PAD_IDX)
        encoder_out = self.encoder(passage_embed, answer_embed, encoder_mask) # N x P x H
        
        dec_out, dec_hiddens = self.decoder(q_emb[:, :-1, :], None) # Teacher Forcing
        
        context_mask = (passage == PAD_IDX)
        self.atten.mask_context(context_mask)
        
        out = self.atten(encoder_out, dec_out)
        out = self.fc(out) # no softmax for training

        return out

    def predict(self, passage, answer):
        """
        passage: 
        """
        question_length = MAX_SHORT_QUESTION_LEN + 2
        passage_embed, answer_embed = self.embedder(passage), self.embedder(answer)
        
        encoder_mask = (answer == PAD_IDX)
        encoder_out = self.encoder(passage_embed, answer_embed, encoder_mask) # N x P x H
        
        context_mask = (passage == PAD_IDX)
        self.atten.mask_context(context_mask)
        
        inputs = torch.ones(passage.shape[0], 1) * START_TKN
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        dec_hidden = None
        prediction = inputs
        for t in range(question_length - 1):
            inputs = self.embedder(inputs.long()) # N x 1 x E

            dec_out, dec_hidden = self.decoder(inputs, dec_hidden)
            out = self.atten(encoder_out, dec_out) # N x 1 x H
            out = self.fc(out) # N x 1 x vocab_size
            probs = self.sm(out.div(self.model_temp))
            
            inputs = torch.multinomial(probs.squeeze(), 1)

            prediction = torch.cat([prediction, inputs], dim=1) # N x Q

        return prediction

    def __call__(self, passage, answer, question=None):
        if question is None:
            return self.predict(passage, answer)
        else:
            return self.forward(passage, answer, question)