from unicodedata import bidirectional
import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_encoder_layers, num_heads):
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads

        self.context_extractor = nn.LSTM(self.embed_dim, self.hidden_size, self.num_encoder_layers, bidirectional=True, batch_first=True)
        self.attention = nn.MultiHeadAttention(2*self.hidden_size, num_heads, batch_first=True)
        self.sm = nn.Softmax(dim=2)

    def forward(self, passage, answer):
        """
        passage: N x P x E
        answer: N x A x E
        """
        bs, P = passage.shape[0], passage.shape[1]
        pass_embed, _ = self.context_extractor(passage) # N x P x 2H
        ans_embed, _ = self.context_extractor(answer)

        # C2Q of multi-head attention?
        # sim = torch.bmm(pass_embed, ans_embed.transpose(1, 2)) # N x P x A
        # a = self.sm(sim)
        # context = torch.bmm(a, ans_embed) # N x P x 2H
        
        context = self.attention(query=pass_embed, key=ans_embed, value=ans_embed) # N x P x 2H
        
        context = context.view(bs, P, 2, self.hidden_size)[:, :, 0, :] # N x P x H
        return context

    def __call__(self, passage, answer):
        return self.forward(passage, answer)

class GlobalAttention(nn.Module):
    
    def __init__(self, embed_dim, hidden_size):
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
        scores.data.masked_fill_(self.context_mask, -float('inf')) # masks out the padding source tokens
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

    def __init__(self, embedding_dim, hidden_size, num_encoder_layers, num_decoder_layers, num_encoder_heads, vocab):
        self.embed_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_heads = num_encoder_heads
        self.vocab_size = len(vocab)

        self.embed_mat = torch.from_numpy(np.load('embeddings.npy')).float()
        self.embedder = nn.Embedding.from_pretrained(self.embed_mat, freeze=True, padding_idx=0)

        self.encoder = Encoder(self.embed_dim, self.hidden_size, self.num_encoder_layers, self.num_encoder_heads)
        self.atten = GlobalAttention(self.embed_dim, self.hidden_size)
        self.decoder = Decoder(self.embed_dim, self.hidden_size, self.num_decoder_layers)

    def forward(self, passage, answer, question):
        passage_embed, answer_embed, q_emb = self.embedder(passage), self.embedder(answer), self.embedder(question)
        encoder_out = self.encoder(passage_embed, answer_embed) # N x P x H
        
        question_length = question.shape[1]
        for i in range(question_length)
