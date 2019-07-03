import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):

    def __init__(self, num_embeddings, embed_size, output_size, activation=F.relu, hidden_size=32, num_layers=5,
                 dropout=0):

        super(Encoder, self).__init__()

        self.activation = activation

        self.embedding = nn.Embedding(num_embeddings, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=True)
        self.linear_layer = nn.Linear(hidden_size*2, output_size)

    def forward(self, input):
        embedding = self.embedding(input).permute(1, 0, -1)  # reshape to seq_len x batch_size x input_size
        output, hidden = self.lstm(embedding)
        # output = self.activation(output)
        # output = self.linear_layer(output[-1])
        return hidden


class Decoder(nn.Module):

    def __init__(self, input_size, output_size, start_token, end_token, activation=torch.tanh,
                 max_iters=200, hidden_size=32, num_layers=5, dropout=0):

        super(Decoder, self).__init__()

        self.max_iters = max_iters
        self.start_token = start_token
        self.end_token = end_token

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout,
                            bidirectional=True)
        self.linear_layer = nn.Linear(hidden_size*2, output_size)
        self.activation = activation

    def forward(self, hidden):
        outputs = []
        x = self.start_token.repeat(1, hidden[0].shape[1], 1)
        for i in range(self.max_iters):
            x, hidden = self.lstm(x, hidden)
            x = self.linear_layer(x)
            x = self.activation(x)
            outputs.append(x)

        result = torch.cat(outputs, dim=0)
        return result

