import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # nn.Embedding holds a Tensor of dim (vocab_size, vector_size), i.e. size of vocab x dim of each vector embedding
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        
        # features: output of encoder -> shape: (batch_size, embed_size)
        # captions: batch of captions -> shape: (batch_size, caption_length)
        
        features = features.unsqueeze(1)
        
        embedding = self.embed(captions[:,:-1])
        embedding = torch.cat((features, embedding), dim=1)
        
        lstm_output, _ = self.lstm(embedding)
        
        output = self.fc(lstm_output)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # initialize the hidden state

        states = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device), 
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        pred_caption = []
        for i in range(max_len):
            
            lstm_output, states = self.lstm(inputs, states)
            
            output = self.fc(lstm_output)
            output = output.squeeze(1)
            
            _, word_index = output.max(1)
            pred_caption.append(word_index.item())
            
            # break if predicted the end token
            if word_index == 1:
                break
            
            # prepare next inputs
            inputs = self.embed(word_index).unsqueeze(1)
            
        return pred_caption