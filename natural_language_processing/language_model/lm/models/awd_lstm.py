import torch
from torch import nn


class WDLSTM(nn.Module):
    """
    A weight dropped LSTM. A simplified version of
    https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
    """

    def __init__(self, dropout, input_size, hidden_size):
        super().__init__()

        # first, an LSTM. num_layers=1 for the reduced hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1)
        self.dropout = dropout

        # set raw weights. We need to set them as new parameters
        # so that the LSTM module still learns
        raw_weights = self.lstm.weight_hh_l0
        del self.lstm._parameters['weight_hh_l0']
        self.lstm.register_parameter('raw_weight_hh_l0', raw_weights)

    def forward(self, x, h0=None):

        # here, we create a mask over some of the weights
        raw_weights = self.lstm.raw_weight_hh_l0
        weight_mask = nn.functional.dropout(raw_weights, p=self.dropout, training=self.training)
        # in case we're on a GPU
        if raw_weights.is_cuda: weight_mask = weight_mask.cuda()

        self.lstm.register_parameter('weight_hh_l0', nn.Parameter(weight_mask))

        output, (hn, cn) = self.lstm(x, h0)

        return output, (hn, cn)


class VDEmbedding(nn.Module):
    """
    An embedding layer with variational dropout
    """
    def __init__(self, dropout, embedding_dim, vocab_size, padding_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = dropout

        # same as above; set the raw weights so that the embeddings can still be learned
        raw_weights = self.embedding.weight
        del self.embedding._parameters['weight']
        self.embedding.register_parameter('raw_weight', raw_weights)

        self.vocab_size = vocab_size

    def forward(self, x):
        raw_weights = self.embedding.raw_weight
        mask = nn.functional.dropout(torch.ones(self.vocab_size),
                                     p=self.dropout, training=self.training).unsqueeze(1)

        masked_weights = mask * raw_weights
        if raw_weights.is_cuda: masked_weights = masked_weights.cuda()

        self.embedding.register_parameter('weight', nn.Parameter(masked_weights))

        return self.embedding(x)


class RecLM(nn.Module):
    """
    Recurrent Language Model
    """
    def __init__(self, embedding_dropout, weight_dropout, embedding_dim, hidden_size,
                 num_layers, vocab_size=60002, padding_idx=0):
        super().__init__()

        # first, the embedding layer with vocabulary-specific dropout
        self.embedding = VDEmbedding(embedding_dropout, embedding_dim, vocab_size, padding_idx)

        for layer in range(num_layers):
            # Independent embedding and hidden size
            if layer == 0:
                wdrnn = WDLSTM(dropout=weight_dropout, input_size=embedding_dim, hidden_size=hidden_size)
            elif layer == (num_layers - 1):
                wdrnn = WDLSTM(dropout=weight_dropout, input_size=hidden_size, hidden_size=embedding_dim)
            else:
                wdrnn = WDLSTM(dropout=weight_dropout, input_size=hidden_size, hidden_size=hidden_size)

            setattr(self, 'wdrnn_{}'.format(layer), wdrnn)

        self.num_layers = num_layers

        self.decoder = nn.Linear(embedding_dim, vocab_size, bias=False)
        # tie weights
        self.decoder.weight = self.embedding.embedding.raw_weight

    def forward(self, x, hidden):

        x = self.embedding(x)
        new_hidden = []
        for layer in range(self.num_layers):
            x, h = getattr(self, 'wdrnn_{}'.format(layer, hidden[layer]))(x)

            if layer == (self.num_layers - 1):
                final_rnn_output = x
                # no need for the cell state here
                final_rnn_hidden = h[0]

            new_hidden.append(h)

        output = self.decoder(x)

        if self.training: return output, new_hidden, (final_rnn_output, final_rnn_hidden)
        else: return output
