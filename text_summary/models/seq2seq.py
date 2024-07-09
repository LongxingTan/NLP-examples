
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, target_seq, max_len, teacher_force_ratio):
        batch_size = encoder_input.size(0)

        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(encoder_input)
        decoder_input = target_seq[:, 0]
        decoder_output = torch.zeros(batch_size, max_len, decoder.hidden_size)

        state = (encoder_hidden, encoder_cell)

        for t in range(1, max_len):
            decoder_output_t, state = self.decoder(decoder_input, state, encoder_output)
            decoder_output[:, t, :] = decoder_output_t
            teacher_force = random.random() < teacher_force_ratio
            top1 = decoder_output_t.max(1)[1]
            decoder_input = target_seq[:, t] if teacher_force else top1

        return decoder_output

    def inference(self, encoder_input, max_len):
        batch_size = encoder_input.size(0)

        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(encoder_input)
        decoder_input = torch.zeros(batch_size, dtype=torch.long)  # Assuming SOS token is 0
        decoder_output = torch.zeros(batch_size, max_len, self.decoder.output_size)

        state = (encoder_hidden, encoder_cell)

        for t in range(1, max_len):
            decoder_output_t, state = self.decoder(decoder_input, state, encoder_output)
            decoder_output[:, t, :] = decoder_output_t
            top1 = decoder_output_t.max(1)[1]
            decoder_input = top1

        return decoder_output


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.rnn(embedded)
        return output, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout)
        self.output_linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x, state, encoder_output):
        x = x.unsqueeze(0)
        embedded = self.embedding(x)
        output, state = self.rnn(embedded, state)
        output = self.output_linear(output.squeeze(0))
        return output, state


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.input_linear = nn.Linear(input_dim, output_dim)
        self.output_linear = nn.Linear(output_dim, output_dim)

    def forward(self, input, source_hidden):
        x = self.input_linear(input)
        scores = (source_hidden * x.unsqueeze(0)).sum(dim=2)
        scores = F.softmax(scores, dim=0)
        x = F.tanh(self.output_linear(torch.cat([x, input], dim=1)))
        return x, scores
