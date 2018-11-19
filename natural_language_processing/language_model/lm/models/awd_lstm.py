import torch
from torch import nn
import math

from .dropout import VariationalDropout

class WDLSTM(nn.Module):
    """
    A weight dropped LSTM. A simplified version of
    https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
    """

    def __init__(self, dropout, input_size, hidden_size):
        super().__init__()
        # first, an LSTM. num_layers=1 for the reduced hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1)
        self.lstm.flatten_parameters = self.fix_weird_pytorch_error

        self.dropout = dropout

        # first, initialize the weights
        self.init_weights(hidden_size)

        # set raw weights. We need to set them as new parameters
        # so that the LSTM module still learns
        raw_weights = self.lstm.weight_hh_l0
        del self.lstm._parameters['weight_hh_l0']
        self.lstm.register_parameter('raw_weight_hh_l0', raw_weights)

    def cleanup(self):
        del self.lstm._parameters['weight_hh_l0']

    def fix_weird_pytorch_error(*args, **kwargs):
        # handling weird pytorch error
        return

    def init_weights(self, hidden_size):
        # first, initialize all lstm weights
        min, max = -1/math.sqrt(hidden_size), 1/math.sqrt(hidden_size)
        for parameter in self.lstm.parameters():
            nn.init.uniform_(parameter, a=min, b=max)

        # set the forget gate biases to 1, to prevent them tending to 0.
        # The forget gate biases are stored in the 1/4 to 1/2 elements of bias_ih_l0
        # and bias_hh_l0
        # http://proceedings.mlr.press/v37/jozefowicz15.pdf

        ih_bias_size = self.lstm.bias_ih_l0.size(0)
        ih_start, ih_end = ih_bias_size//4, ih_bias_size//2
        self.lstm.bias_ih_l0.data[ih_start:ih_end].fill_(1)

        hh_bias_size = self.lstm.bias_hh_l0.size(0)
        hh_start, hh_end = hh_bias_size // 4, hh_bias_size // 2
        self.lstm.bias_hh_l0.data[hh_start:hh_end].fill_(1)

    def forward(self, x, h0=None):

        # here, we create a mask over some of the weights
        raw_weights = self.lstm.raw_weight_hh_l0
        weight_mask = nn.functional.dropout(raw_weights, p=self.dropout, training=self.training)
        # in case we're on a GPU
        if raw_weights.is_cuda: weight_mask = weight_mask.cuda()

        self.lstm.register_parameter('weight_hh_l0', nn.Parameter(weight_mask))

        # Passing h0=None initializes the hidden layer to all 0s,
        # so this is the equivalent of passing a 0 tensor.
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

        # initialize the weights
        self.init_weights()

        # same as for the WDLSTM; set the raw weights so that the embeddings can still be learned
        raw_weights = self.embedding.weight
        del self.embedding._parameters['weight']
        self.embedding.register_parameter('raw_weight', raw_weights)

        self.vocab_size = vocab_size

    def cleanup(self):
        del self.embedding._parameters['weight']

    def init_weights(self):
        # embedding weights are uniformly initialized between -0.1 and 0.1
        nn.init.uniform_(self.embedding.weight, a=-0.1, b=0.1)

    def forward(self, x):
        raw_weights = self.embedding.raw_weight
        mask = nn.functional.dropout(torch.ones(self.vocab_size),
                                     p=self.dropout, training=self.training).unsqueeze(1)
        # in case we're on a GPU
        if raw_weights.is_cuda: mask = mask.cuda()

        masked_weights = mask * raw_weights

        self.embedding.register_parameter('weight', nn.Parameter(masked_weights))

        return self.embedding(x)


class ARTAR(nn.Module):
    """
    Activation Regularization and Temporal Activation Regularization

    Arguments:
        alpha: scaling coefficient for the AR regularization
            (default: 2)
        beta: scaling coefficient for the TAR regularization
            (default: 1)
    Defaults taken from the awd_lstm paper
    """
    def __init__(self, alpha=2, beta=1):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

    def forward(self, hidden):
        ar = torch.norm(hidden)

        diff = hidden[:, :-1, :] - hidden[:, 1:, :]
        tar = torch.norm(diff)

        return (self.alpha * ar) + (self.beta * tar)


class RecLM(nn.Module):
    """
    Recurrent Language Model

    Default values taken from the awd_lstm paper
    """
    def __init__(self, word_dropout=0.4, rnn_weight_dropout=0.3, var_dropout_emb=0.1, var_dropout_rnn=0.4,
                 embedding_dim=400, hidden_size=1150, num_layers=3, vocab_size=30002, padding_idx=0,
                 finetuning=False):
        super().__init__()

        self.finetuning = finetuning
        # first, the embedding layer with vocabulary-specific dropout
        self.embedding = VDEmbedding(word_dropout, embedding_dim, vocab_size, padding_idx)

        self.rnns = nn.ModuleList([WDLSTM(dropout=rnn_weight_dropout,
                                          input_size=embedding_dim if i == 0 else hidden_size,
                                          hidden_size=hidden_size if i != (num_layers - 1) else embedding_dim)
                                   for i in range(num_layers)])

        if not finetuning:
            self.decoder = nn.Linear(embedding_dim, vocab_size)
            self.init_weights()

        # finally, variational dropout
        self.emb_drop = VariationalDropout(p=var_dropout_emb)
        self.final_rnn_drop = VariationalDropout(p=var_dropout_rnn)

    def cleanup(self):
        self.embedding.cleanup()
        for wdrnn in self.rnns:
            wdrnn.cleanup()

    def init_weights(self):
        self.decoder.weight = self.embedding.embedding.raw_weight
        self.decoder.bias.data.fill_(0)

    def forward(self, x):

        x = self.emb_drop(self.embedding(x))
        for i, wdrnn in enumerate(self.rnns):
            x, hidden = wdrnn(x)
            if i == (len(self.rnns) - 1):
                # no need for the cell state here
                final_rnn_hidden = hidden[0]
        if not self.finetuning:
            # we only want the last output to be decoded
            output = self.decoder(x)
            if self.training: return output, final_rnn_hidden
            else: return output
        else:
            output = self.final_rnn_drop(x)
            if self.training: return output, final_rnn_hidden
            else: return output
