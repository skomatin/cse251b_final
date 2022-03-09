import torch
import torch.nn as nn
import torchvision
import numpy as np

class base_LSTM(nn.Module):
    
    def __init__(self, hidden_size, embedding_size, num_layers, vocab_size, model_temp):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.model_temp = model_temp

        self.encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.ffn = nn.Conv2d(in_channels=2*self.hidden_size, out_channels=self.embedding_size, kernel_size=1)
        self.word_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        
        self.decoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

    def forward(self, inp, hidden_state=None):
        """
        Generates raw logits over vocabulary from given input at timestep t
        inp: N x 1 x embedding_size
        out: N x 1 x vocab_size
        """
        if hidden_state is None:
            out, hidden_state = self.decoder(inp)
        else:
            out, hidden_state = self.decoder(inp, hidden_state)
        
        out = self.fc(out)
        return out, hidden_state

    def predict(self, passage, answer, question_length):

        passage_length = passage.shape[1]
        answer_length = answer.shape[1]

        linked_input = np.concatenate((passage, answer), axis=1)
        linked_embedded = self.word_embedding(linked_input)
        encoded_inp = self.encoder(linked_embedded)
        temp = self.ffn(encoded_inp)
        pool = nn.AvgPool2d((1, passage_length+answer_length))
        inp = pool(temp)
        
        hidden_state = None
        prediction = None
        for i in range(question_length):
            out, hidden_state = self.forward(inp, hidden_state) # N x 1 x vocab_size
            
            probs = nn.Softmax(dim=2)(out.div(self.model_temp)).squeeze() # N x vocab_size
            word = torch.multinomial(probs, 1) # N x 1

            if i == 0:
                prediction = word
            else:
                prediction = torch.cat([prediction, word], axis=1) # N x L
            
            inp = self.embed_word(word.long()) # N x 1 x 300
        
        return prediction

    def __call__(self, inp, hidden_state=None):
        return self.forward(inp, hidden_state)
        
        