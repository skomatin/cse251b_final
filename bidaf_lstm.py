import torch
import torch.nn as nn

class LSTMContextLayer(nn.Module):
    """Extracts contextual information from inputs"""
    def __init__(self, embedding_dim, hidden_size, num_layers):
        super().__init__()
        
        self.pass_context = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.ans_context = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, passage, answer):
        """
        passage - N x P x E
        answer - N X A X E
        """

        out_pass, _ = self.pass_context(passage) # N x P x 2*H
        out_ans, _ = self.ans_context(answer) # N x A x 2*H

        return out_pass, out_ans

    def __call__(self, passage, answer):
        return self.forward(passage, answer)

class AttentionFlowLayer(nn.Module):
    """
    Computes C2Q and Q2C as per
    https://arxiv.org/pdf/1611.01603.pdf
    https://towardsdatascience.com/the-definitive-guide-to-bidaf-part-2-word-embedding-character-embedding-and-contextual-c151fc4f05bb
    https://towardsdatascience.com/the-definitive-guide-to-bidaf-part-3-attention-92352bbdcb07
    """
    def __init__(self, embedding_dim, hidden_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.weights = nn.Linear(6*self.hidden_size, 1, bias=False)

    def forward(self, H, U):
        """
        H: N x P x 2d -- passage representation
        U: N x A x 2d -- answer representation

        G (output): N x P x 2d
        Implementation based off of
        https://github.com/jojonki/BiDAF/blob/master/layers/bidaf.py
        """
        context = H.unsqueeze(2) # N x P x 1 x 2d
        ans = U.unsqueeze(1) # N x 1 x A x 2d

        cast_shape = (H.shape[0], H.shape[1], U.shape[1], 2*self.hidden_size) # N x P x A x 2d
        context = context.expand(cast_shape)
        ans = ans.expand(cast_shape)
        
        # Similarity Matrix Passage and Answer
        prod = torch.mul(context, ans)
        vec = torch.cat([context, ans, prod], axis=3) # N x P x A x 6d
        sim = self.weights(vec).squeeze() # N x P x A

        # C2Q - which query (ie. answer) words matter most to each context (ie. passage) word
        a = nn.Softmax(dim=2)(sim) # N x P x A
        U_tilda = torch.bmm(a, U) # N x P x 2d

        # Q2C
        b = torch.max(sim, dim=2)[0].unsqueeze(1) # N x 1 x P
        H_tilda = torch.bmm(b, H).tile((1, H.shape[1], 1)) # N x P x 2d

        # Merge to form G
        G = torch.cat([H, U_tilda, torch.mul(H, U_tilda), torch.mul(H, H_tilda)], dim=2) # N x P x 8d
        return G

    def __call__(self, H, U):
        return self.forward(H, U)

class AttentionFlowLSTMEncoder(nn.Module):

    def __init__(self, embedding_dim, hidden_size, num_layers, vocab):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab = vocab
        self.embedder = nn.Embedding(num_embeddings = len(self.vocab), embedding_dim=embedding_dim, padding_idx=0)
        self.context_layer = LSTMContextLayer(self.embedding_dim, self.hidden_size, self.num_layers)
        self.attention_flow_layer = AttentionFlowLayer(self.embedding_dim, self.hidden_size)

        self.encoder = nn.Sequential(
            nn.LSTM(8*self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True),
            nn.Linear(2*self.hidden_size, self.embedding_dim)
        )

    def embed(self, words):
        """words is N x L"""
        return self.embedder(words) # N X L X embedding

    def forward(self, passage, answer):
        """
        passage: embedded passage N x P x E
        answer: embedded answer N x A x E
        """

        H, U = self.context_layer(passage, answer) # N x P x 2H, N x A x 2H
        G = self.attention_flow_layer(H, U) # N x P x 8H
        encoding, _ = self.encoder(G) # N x P x 2H

        return encoding

    def __call__(self, passage, answer):
        return self.forward(passage, answer)

class LSTMDecoder(nn.Module):
    """Decoder to produce sequential output as question"""

    def __init__(self, embedding_dim, hidden_size, vocab, question_length):
        """
        embedding_dim: dimension of word embedding
        hidden_size: hidden state dimension for decoder
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.question_length = question_length

        self.decoder = nn.Sequential(
            nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True),
            nn.Linear(self.hidden_size, len(self.vocab))
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, encoded_inputs, embedded_targets):
        """
        Generates sequential output using teacher forcing
        
        encoded_inputs: N x L x E
        embedded_targets: N x L x E
        """
        assert encoded_inputs.shape[1] == self.question_length
        
        decoder_inputs = torch.cat([encoded_inputs, embedded_targets[:, :-1, :]], dim=1)
        sequence = self.decoder(decoder_inputs) # N x L x vocab_size

        return sequence

    def __call__(self, encoded_inputs, embedded_targets):
        
        return self.forward(encoded_inputs, embedded_targets)

class BiDAF_LSTMNet(nn.Module):