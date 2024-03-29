import torch
import torch.nn as nn
import torchvision
from constants import *

class base_LSTM(nn.Module):
    
    def __init__(self, hidden_size, embedding_size, num_layers, vocab, model_temp):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.vocab_size = len(vocab)
        self.model_temp = model_temp
        self.passage_length = MAX_PASSAGE_LEN+2
        self.answer_length = MAX_ANSWER_LEN+2
        self.question_length = MAX_QUESTION_LEN+2

        self.encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.fcn = nn.Linear(in_features=2*self.hidden_size, out_features=self.embedding_size)
#         self.ffn = nn.Conv2d(in_channels=2*self.hidden_size, out_channels=self.embedding_size, kernel_size=1)
        self.word_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)

#         self.pool = nn.AvgPool2d((1, self.passage_length+self.answer_length))
        # self.pool = nn.AvgPool1d(kernel_size=self.passage_length + self.answer_length)
        
        self.decoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

    def forward(self, passage, answer, question):
        linked_input = torch.cat((passage, answer), dim=1)
        linked_embedded = self.word_embedding(linked_input) #(batch_size, passage_size+answer_size, embedding_size)
        embedded_passage = torch.split(linked_embedded, [self.passage_length, self.answer_length], dim=1)[0]
        embedded_answer = torch.split(linked_embedded, [self.passage_length, self.answer_length], dim=1)[1]
        encoded_passage = self.encoder(embedded_passage)[0] #tuple
        encoded_answer = self.encoder(embedded_answer)[0]

        linked_encoded = torch.cat((encoded_passage, encoded_answer), dim=1) #(batch, num_words, 2*hidden_size)
        temp = self.fcn(linked_encoded) #(batch, num_words, embedding_size)
        inp_pa = torch.mean(temp, dim=1, keepdim=True)

        inp_q = torch.split(question, [self.question_length-1, 1], dim=1)[0]
        embedded_q = linked_embedded = self.word_embedding(inp_q)
        inp = torch.cat((inp_pa, embedded_q), dim=1)

        out, _ = self.decoder(inp)
        out = self.fc(out)

        return out

    def predict(self, passage, answer):

        linked_input = torch.cat((passage, answer), dim=1)
        linked_embedded = self.word_embedding(linked_input)
        embedded_passage = torch.split(linked_embedded, [self.passage_length, self.answer_length], dim=1)[0]
        embedded_answer = torch.split(linked_embedded, [self.passage_length, self.answer_length], dim=1)[1]
        encoded_passage = self.encoder(embedded_passage)[0]
        encoded_answer = self.encoder(embedded_answer)[0]

        linked_encoded = torch.cat((encoded_passage, encoded_answer), dim=1)
        temp = self.fcn(linked_encoded)
        inp = torch.mean(temp, dim=1, keepdim=True)
        
        hidden_state = None
        prediction = None
        for i in range(self.question_length):

            if hidden_state is None:
                out, hidden_state = self.decoder(inp)
            else:
                out, hidden_state = self.decoder(inp, hidden_state)
        
            out = self.fc(out)
            
            probs = nn.Softmax(dim=2)(out.div(self.model_temp)).squeeze() # N x vocab_size
            word = torch.multinomial(probs, 1) # N x 1

            if i == 0:
                prediction = word
            else:
                prediction = torch.cat([prediction, word], dim=1) # N x L
            
            inp = self.word_embedding(word.long()) # N x 1 x 300
        
        return prediction

    def __call__(self, passage, answer, question):
        return self.forward(passage, answer, question)
        
        
