from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from .awd_lstm import VDEmbedding
from .dropout import VariationalDropout


class WDConv(nn.Module):
    """
    A variationally weight dropped ConvNet.
    To be used in the TCN convolutional block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation,
                 padding, dropout):
        super().__init__()

        # first, a conv layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding,
                               dilation)
        self.dropout = dropout

        self.init_weights()

        raw_weights = self.conv.weight
        del self.conv._parameters['weight']
        # assigning it to the module, so that we can call weight_norm on it
        self.register_parameter('raw_weight', raw_weights)

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def cleanup(self):
        """Removes the weight parameter, so that the state dict can be used
        to load other models
        """
        del self.conv._parameters['weight']

    def forward(self, x):
        raw_weights = self.raw_weight
        weight_mask = nn.functional.dropout(raw_weights, p=self.dropout, training=self.training)

        # in case we're on a GPU
        if raw_weights.is_cuda: weight_mask = weight_mask.cuda()

        self.conv.register_parameter('weight', nn.Parameter(weight_mask))
        # this makes sure we only look into the past
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    A weight dropped residual TCN Block
    """

    def __init__(self, num_layers, in_channels, hidden_channels, kernel_size, stride, dilation,
                 dropout):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.wd_convs = nn.ModuleList([weight_norm(WDConv(in_channels if i == 1 else hidden_channels,
                                       hidden_channels if i != num_layers else in_channels,
                                       kernel_size, stride=stride, padding=self.padding,
                                       dilation=dilation, dropout=dropout),
                                       name='raw_weight') for i in range(1, num_layers + 1)])

    def cleanup(self):
        for conv in self.wd_convs:
            conv.cleanup()

    def forward(self, x):

        for i, conv in enumerate(self.wd_convs):
            if i == 0:
                out = F.relu(conv(x)[:, :, :-self.padding])
            else:
                out = F.relu(conv(out)[:, :, :-self.padding])
        return F.relu(out + x)


class ConvLM(nn.Module):
    """
    Arguments:
        num_blocks: number of TCN blocks
        num_layers: number of convolutional layers per TCN block
        embedding_dim: the input embedding size.
        hidden_channels: the output embedding sizes of the interim layers. Similar to hidden
            size in the awd_lstm
        kernel_size: float. Kernel sizes for the convolutional layers in the block
        conv_dropout: float. DropConnect values for the convolutional layers in the block.
        embedding_dropout: float. Dropout for the embedding layers
    """

    def __init__(self, num_blocks=3, num_layers=3, embedding_dim=400, hidden_channels=1150, kernel_size=2,
                 conv_dropout=0.2, embedding_dropout=0.1, var_dropout_emb=0.1, vocab_size=30002, padding_idx=0,
                 finetuning=False):
        super().__init__()

        self.finetuning = finetuning
        self.embedding = VDEmbedding(embedding_dropout, embedding_dim, vocab_size, padding_idx)

        self.convblocks = nn.ModuleList([TCNBlock(num_layers, embedding_dim, hidden_channels, kernel_size,
                                                  stride=1, dilation=2 ** block, dropout=conv_dropout)
                                         for block in range(num_blocks)])
        if not finetuning:
            self.decoder = nn.Linear(embedding_dim, vocab_size)
            self.init_weights()

        # finally, dropouts
        self.emb_drop = VariationalDropout(p=var_dropout_emb)

    def init_weights(self):
        # tie weights
        self.decoder.weight = self.embedding.embedding.raw_weight
        self.decoder.bias.data.fill_(0)

    def cleanup(self):
        self.embedding.cleanup()
        for convblock in self.convblocks:
            convblock.cleanup()

    def forward(self, x):

        x = self.emb_drop(self.embedding(x))
        x = x.permute(0, 2, 1).contiguous()
        for convblock in self.convblocks:
            x = convblock(x)
        x = x.permute(0, 2, 1).contiguous()
        # the output is flattened so that it can be
        # passed to cross entropy loss
        if not self.finetuning: return self.decoder(x)
        else: return x
