import torch
import torch.nn as nn
import torchvision

class EncoderDecoderLSTM(nn.Module):
    
    def __init__(self, hidden_size, embedding_size, num_layers, vocab_size, model_temp):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.model_temp = model_temp
        
        # Keep AdaptiveAvgPool2D? -- 2048 x 8 x 8 before Avg, 2048 x 1 after
        resnet = torchvision.models.resnet50(pretrained=True)
        mods = list(resnet.children())[:-1]
        # mods = list(resnet.children())[:-2]

        self.encoder = nn.Sequential(*mods)
        for p in self.encoder.parameters():
            p.requires_grad = False
            
        self.image_embedding = nn.Linear(in_features=2048, out_features=self.embedding_size)
        # self.image_embedding = nn.Linear(in_features=2048*8*8, out_features=self.embedding_size)
        self.word_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        
        self.decoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        
    def embed_image(self, img):
        """
        Embeds a batch of images
        img: N x 3 x H x W
        out: N x 1 x embedding_size
        """
        out = self.encoder(img) # N x 2048 x 1 x 1
        out = out.flatten(start_dim=1, end_dim=-1).unsqueeze(1) # N x 1 x 2048
        out = self.image_embedding(out) # N x 1 x embedding_size
        return out
    
    def embed_word(self, word):
        """
        Embeds a batch of words
        word: N x L
        out: N x L x embedding_size
        """
        out = self.word_embedding(word)
        return out
    
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

    def predict(self, img, caption_length):
        """
        Generates a predicted caption for a given set of images
        img: N x 3 x H x W
        prediction: N x L
        """
        inp = self.embed_image(img) # N x 1 x embedding_size
        
        hidden_state = None
        prediction = None
        for i in range(caption_length):
            out, hidden_state = self.forward(inp, hidden_state) # N x 1 x vocab_size
            
            probs = nn.Softmax(dim=2)(out.div(self.model_temp)).squeeze() # N x vocab_size
            word = torch.multinomial(probs, 1) # N x 1

            if i == 0:
                prediction = word
            else:
                prediction = torch.cat([prediction, word], axis=1) # N x L
            
            inp = self.embed_word(word.long()) # N x 1 x embedding_size
        
        return prediction

    def __call__(self, inp, hidden_state=None):
        return self.forward(inp, hidden_state)
        
        