'''
This program defines a class that creates an LSTM model and supports feeding input into the LSTM model.
'''
import torch
from torch import nn
import torch.autograd as autograd

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, embedding_matrix,\
        hidden_dim, n_layers, input_len, pretrain=False):
        # Setting up layers
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_len = input_len
        
        # Setting up embeddings
        if pretrain:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.init_weights()
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True) # Define LSTM model
        self.dropout = nn.Dropout(0.3) 
        self.pool = nn.MaxPool1d(self.input_len)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def init_weights(self):
        # Intializing weights
        self.embedding.weight.data.uniform_(-0.1, 0.1)
    
    def _init_hidden(self, batch_size):
        # Initializing hidden state and cell state
        return(autograd.Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim)).to(device), autograd.Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim)).to(device))

    def forward(self, x):
        # Feeding input into LSTM model
        batch_size = x.size(0)
        hidden_cell = self._init_hidden(batch_size)
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds, hidden_cell)
        lstm_out = lstm_out.permute(0,2,1) # Permute the position of hidden_dim and seq_len
        out = self.pool(lstm_out) # Pool the output to reduce seq_len dim
        out = out.view(out.size(0),-1)
        out = self.dropout(out)
        out = self.fc(out) # Feed into linear layer
        out = self.sigmoid(out)
        out = out[:,0]
        return out
        