import torch
import numpy as np
import torch.nn as nn

class ContextLayer(nn.Module):
    """Extracts contextual information from inputs"""
    def __init__(self, embedding_dim, hidden_size, num_layers):
        super().__init__()
        
        self.ans_context = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.pass_context = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, passage, answer):
        """
        passage - N x P x E
        answer - N X A X E
        """

        out_pass, _ = self.pass_context(passage) # N x P x 2*H
        out_ans, _ = self.ans_context(answer) # N x A x 2*H

        return out_pass, out_ans

class AttentionFlow(nn.Module):
    """
    Computes C2Q and Q2C as per
    https://arxiv.org/pdf/1611.01603.pdf
    https://towardsdatascience.com/the-definitive-guide-to-bidaf-part-2-word-embedding-character-embedding-and-contextual-c151fc4f05bb
    https://towardsdatascience.com/the-definitive-guide-to-bidaf-part-3-attention-92352bbdcb07
    """
    def__init__(self, embedding_dim, hidden_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.weights = nn.Linear(6*self.hidden_size, 1, bias=False)
        
    def forward(self, H, U):
        """
        H: N x P x 2d -- passage representation
        U: N x A x 2d -- answer representation

        Implementation based off of
        https://github.com/jojonki/BiDAF/blob/master/layers/bidaf.py
        """
        context = H.unsqueeze(2) # N x P x 1 x 2d
        ans = U.unsqueeze(1) # N x 1 x A x 2d

        cast_shape = (H.shape[0], H.shape[1], U.shape[1], 2*self.hidden_size) # N x P x A x 2d
        context = context.expand(cast_shape)
        ans = ans.expand(cast_shape)
        
        prod = torch.mul(context, ans)
        vec = torch.cat([context, ans, prod], axis=3) # N x P x A x 6d
        sim = self.weights(vec).view(H.shape[0], H.shape[1], U.shape[1]) # N x P x A

        return sim



class BiDirectionalEncoder(nn.Module):

    def __init__(self, embedding_dim, hidden_size, num_layers, vocab):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.embedder = nn.Embedding(num_embeddings = len(self.vocab), embedding_dim=embedding_dim, padding_idx=0)
        self.context_layer = ContextLayer(self.embedding_dim, hidden_size, num_layers)

    def embed(self, words):
        """words is N x L"""
        return self.embedder(words) # N X L X embedding

    