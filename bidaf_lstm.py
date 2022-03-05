import torch
import torch.nn as nn
import constants

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
        self.context_layer = LSTMContextLayer(self.embedding_dim, self.hidden_size, self.num_layers)
        self.attention_flow_layer = AttentionFlowLayer(self.embedding_dim, self.hidden_size)

        self.encoder = nn.LSTM(8*self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*self.hidden_size, self.embedding_dim)

    def forward(self, passage, answer):
        """
        passage: embedded passage N x P x E
        answer: embedded answer N x A x E

        encoding: encoded representation of passage-answer N x 1 x E
        """

        H, U = self.context_layer(passage, answer) # N x P x 2H, N x A x 2H
        G = self.attention_flow_layer(H, U) # N x P x 8H
        encoding, _ = self.encoder(G) # N x P x 2H
        encoding = torch.mean(encoding, dim=1, keepdim=True) # N x 1 x 2H
        encoding = self.fc(encoding) # N x 1 x E
        return encoding

    def __call__(self, passage, answer):
        return self.forward(passage, answer)

class LSTMDecoder(nn.Module):
    """Decoder to produce sequential output as question"""

    def __init__(self, embedding_dim, hidden_size, num_layers, vocab, question_length):
        """
        embedding_dim: dimension of word embedding
        hidden_size: hidden state dimension for decoder
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab = vocab
        self.question_length = question_length

        self.decoder = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, len(self.vocab))
        self.softmax = nn.Softmax(dim=2)

    def forward(self, encoded_inputs, embedded_targets=None, embedder=None, temperature=1.0):
        """
        Generates sequential output
        
        encoded_inputs: N x 1 x E
        embedded_targets: N x Q x E
        embedded_targets: supply targets for teacher forcing, otherwise, a single output is produced
        """
        
        if embedded_targets is not None:        
            assert embedded_targets.shape[1] == self.question_length
            decoder_inputs = torch.cat([encoded_inputs, embedded_targets[:, :-1, :]], dim=1)
            sequence, _ = self.decoder(decoder_inputs) # N x Q x hidden_size
            sequence = self.fc(sequence) # N x Q x vocab_size
            return sequence
        else:
            with torch.no_grad():
                assert embedder is not None
                sequence = None
                hidden_state = None
                inputs = encoded_inputs
                for i in range(self.question_length):
                    out, hidden_state = self.decoder(inputs, hidden_state) # N x 1 x H
                    out = self.fc(out) # N x 1 x vocab_size
                    probs = self.softmax(out.div(temperature))
                    word = torch.multinomial(probs.squeeze(), 1) # N x 1

                    if i == 0:
                        sequence = word
                    else:
                        sequence = torch.cat([sequence, word], dim=1)

                    inputs = embedder(word.long()) # N x 1 x E
                    
                return sequence # N x Q
    
    def __call__(self, encoded_inputs, embedded_targets, embedder=None, temperature=1.0):
        
        return self.forward(encoded_inputs, embedded_targets, embedder, temperature)

class BiDAF_LSTMNet(nn.Module):

    def __init__(self, embedding_dim, hidden_size, num_layers, vocab, question_length, temperature=1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab = vocab
        self.question_length = question_length
        self.temperature = temperature
        
        self.embedder = nn.Embedding(num_embeddings = len(self.vocab), embedding_dim=embedding_dim, padding_idx=0)
        self.encoder = AttentionFlowLSTMEncoder(self.embedding_dim, self.hidden_size, self.num_layers, self.vocab)
        self.decoder = LSTMDecoder(self.embedding_dim, self.hidden_size, self.num_layers, self.vocab, self.question_length)
        self.softmax = nn.Softmax(dim=2)

    def embed(self, words):
        """words is N x L"""
        return self.embedder(words) # N X L X embedding

    def forward(self, passage, answer, question):
        """
        passage: N x P
        answer: N x A
        question: N x Q
        """
        P = passage.shape[1] 
        A = answer.shape[1]
        Q = question.shape[1]
        
        assert P == constants.MAX_PASSAGE_LEN + 2
        assert A == constants.MAX_ANSWER_LEN + 2
        assert Q == constants.MAX_QUESTION_LEN + 2
        
        pass_ans_qembed = self.embed(torch.cat([passage, answer, question], dim=1))
        passage, answer, q_embed = pass_ans_qembed[:, :P, :], pass_ans_qembed[:, P:P+A, :], pass_ans_qembed[:, P+A:, :]

        encoded = self.encoder(passage, answer)
        sequence = self.decoder(encoded, q_embed) # teacher forced

        return sequence # N x Q x vocab_size

    def predict(self, passage, answer):
        """
        Generates question word-by-word

        Should only be used in evaluation mode
        """
        with torch.no_grad():
            pass_ans = self.embed(torch.cat([passage, answer], dim=1))
            passage, answer = pass_ans[:, :passage.shape[1], :], pass_ans[:, passage.shape[1]:, :]
            encoded = self.encoder(passage, answer) # N x 1 x E
            sequence = self.decoder(encoded, embedded_targets=None, embedder=self.embedder, temperature=self.temperature) # N x Q
            return sequence