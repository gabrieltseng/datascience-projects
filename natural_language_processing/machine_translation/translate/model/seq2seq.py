import torch
from torch import nn

import random

from ..data import BOS_TOKEN, EOS_TOKEN


class Attention(nn.Module):
    def __init__(self, rnn_hidden_size):
        super().__init__()
        self.hidden_linear = nn.Linear(in_features=rnn_hidden_size, out_features=rnn_hidden_size)
        self.total_linear = nn.Linear(in_features=rnn_hidden_size, out_features=1)

    def forward(self, transformed_encodings, hidden):
        """
        Assumes the encodings have already been put through a linear layer,
        since there is no need to do this for every loop in the decoder
        """
        x = torch.tanh(transformed_encodings + self.hidden_linear(hidden))
        # first dimension is the sequences
        return nn.functional.softmax(self.total_linear(x), dim=1)


class FrenchToEnglish(nn.Module):

    def __init__(self, fr_embedding_path, en_embedding_path, fr_dict, en_dict, max_en_length,
                 rnn_hidden_size=256, embedding_size=300, embedding_mean=0, embedding_std=0.3,
                 encoder_dropout=0.15, decoder_dropout=0.35, bidirectional=True, forcing_probability=0,
                 attention=False):
        super().__init__()

        self.bidirectional = bidirectional
        self.fr_embedding = self.get_embedding(fr_embedding_path, fr_dict, embedding_size,
                                               embedding_mean, embedding_std)
        self.en_embedding = self.get_embedding(en_embedding_path, en_dict, embedding_size,
                                               embedding_mean, embedding_std)

        self.encoder = nn.GRU(input_size=embedding_size, hidden_size=rnn_hidden_size, num_layers=2,
                              batch_first=True, dropout=encoder_dropout, bidirectional=bidirectional)

        # a linear transformation from the encoder to the decoder 2)
        self.hidden_transformer = nn.Linear(rnn_hidden_size * 2 if bidirectional else rnn_hidden_size,
                                            rnn_hidden_size)
        # add attention
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(rnn_hidden_size=rnn_hidden_size)
            self.rnn_transformer = nn.Linear((rnn_hidden_size * 2 if bidirectional else rnn_hidden_size) +
                                             embedding_size, embedding_size)
            self.encoding_transformer = nn.Linear(rnn_hidden_size * 2 if bidirectional else rnn_hidden_size,
                                                  rnn_hidden_size)

        self.decoder = nn.GRU(input_size=embedding_size, hidden_size=rnn_hidden_size, num_layers=2,
                              batch_first=True)
        self.decoder_dropout = nn.Dropout(p=decoder_dropout)
        self.to_vocab = nn.Linear(rnn_hidden_size, len(en_dict))

        # the max length of the output
        self.max_length = max_en_length
        self.en_bos = en_dict[BOS_TOKEN]
        self.en_eos = en_dict[EOS_TOKEN]

        # although this can be set here, ideally it would be controlled by the scheduler
        self.forcing_probability = forcing_probability

    def get_embedding(self, embedding_path, language_dict, embedding_size=300, mean=0, std=0.3):
        """
        Load a language embedding from fasttext vectors

        Arguments:
            embedding_path: a pathlib.Path of the embeddings' path
            language_dict: word to int
            embedding_size: the embedding size
            mean and std are the pre-trained embedding values, so that the
            randomly initialized layers are also handled
        """
        embedding = nn.Embedding(num_embeddings=len(language_dict),
                                 embedding_dim=embedding_size,
                                 padding_idx=language_dict['_pad_'])

        # aligns with what we will fill it up as
        nn.init.normal_(embedding.weight, mean=mean, std=std)

        with torch.no_grad():
            with embedding_path.open('r', encoding='utf-8', newline='\n', errors='ignore') as f:
                for line in f:
                    tokens = line.rstrip().split(' ')
                    if tokens[0] in language_dict:
                        embedding.weight[language_dict[tokens[0]]] = torch.tensor(list(map(float, tokens[1:])))
        return embedding

    def forward(self, fr, en=None):
        """
        fr: the questions to be translated
        en: if not None, will be used with teacher forcing
        """
        # first, get the embeddings for the french input questions
        batch_size = fr.shape[0]
        fr_emb = self.fr_embedding(fr)

        # we only care about the hidden output of the encoder
        encoding, hidden = self.encoder(fr_emb)
        if self.bidirectional:
            d1, _, d3 = hidden.shape
            # if bidirectional, d1 will definitely be divisible by 2
            hidden = hidden.view(int(d1 / 2), batch_size, int(d3 * 2))
        if self.attention:
            transformed_encoding = self.encoding_transformer(encoding)
        hidden = self.hidden_transformer(hidden)
        # generate a [batch_size, 1] dimensional tensor of the beginning of sentence tokens
        base = torch.ones(batch_size).long().unsqueeze(1)
        if self.en_embedding.weight.is_cuda: base = base.cuda()

        seq_tensor = self.decoder_dropout(self.en_embedding(base * self.en_bos))
        en_questions = []
        for i in range(self.max_length):
            if self.attention:
                attention = self.attention_layer(transformed_encoding,
                                                 hidden[-1].unsqueeze(0).transpose(1, 0))
                weighted_inputs = (attention * encoding).sum(1).unsqueeze(1)
                seq_tensor = self.rnn_transformer(torch.cat([weighted_inputs, seq_tensor], dim=-1))
            output, hidden = self.decoder(seq_tensor, hidden)
            words = self.to_vocab(output)
            en_questions.append(words)
            # check if we should use forced teaching now
            if (en is not None) and (self.forcing_probability > 0):
                if random.random() < self.forcing_probability:
                    if i < en.shape[1]:
                        selected_words = en[:, i].unsqueeze(1)
                    else:
                        return torch.cat(en_questions, dim=1)
                else:
                    selected_words = words.argmax(dim=-1)
            else:
                selected_words = words.argmax(dim=-1)
            # check we are not all at an end of sentence token
            if torch.eq(selected_words, self.en_eos).all():
                return torch.cat(en_questions, dim=1)
            seq_tensor = self.decoder_dropout(self.en_embedding(selected_words))
        return torch.cat(en_questions, dim=1)
