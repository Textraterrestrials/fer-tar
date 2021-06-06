import torch
from torch import nn
from torch.nn import functional as F


class AvgPoolingModel(nn.Module):

    def __init__(self, embedding, embedding_size=300):
        super(AvgPoolingModel, self).__init__()
        self.embedding = embedding
        self.linear1 = nn.Linear(embedding_size, 150)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(150, 150)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(150, 1)

    def forward(self, batch, lens):
        batch_embedded = self.embedding(batch)
        # sum across all words
        batch_averaged = batch_embedded.sum(dim=1)
        # divide each result by the number of words
        for i in range(len(batch)):
            batch_averaged[i] /= lens[i]
        h1 = self.relu1(self.linear1(batch_averaged))
        h2 = self.relu2(self.linear2(h1))
        h3 = self.linear3(h2)
        return torch.reshape(h3, (h3.shape[0],))


class LSTMModel(nn.Module):
    """Hardkodirano je da je bidirectional i batch_first."""

    def __init__(self, embedding, input_size=300, hidden_size=150, num_layers=2, dropout=0.9):
        super(LSTMModel, self).__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout
        )
        self.linear = nn.Linear(2 * hidden_size, 1)
        # self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(100, 1)

    def forward(self, s0_batch, s1_batch, lens0, lens1):
        # batch.shape = (batch_size, seq_len, emb_size)
        s0_batch = self.embedding(s0_batch)
        s1_batch = self.embedding(s1_batch)
        # output.shape = (batch_size, seq_len, hidden_size)
        # hn.shape = cn.shape = (num_layers, batch_size, hidden_size)
        output0, _ = self.lstm(s0_batch)
        output1, _ = self.lstm(s1_batch)
        # samo zadnji layer (-1)
        last_hidden_state_0 = output0[:, -1]
        last_hidden_state_1 = output1[:, -1]
        score0 = self.linear(last_hidden_state_0)
        score1 = self.linear(last_hidden_state_1)
        e0 = torch.exp(score0)
        e1 = torch.exp(score1)
        output = e0 / (e0 + e1)
        return output


class AttentionBiLSTM(nn.Module):
    def __init__(self, embedding, **config):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.gpu = config['gpu']

        self.embedding = embedding
        # self.embedding = nn.Embedding(vocab_size, config['emb_dim'])
        # if vec is not None:
        #    self.embedding.weight.data.copy_(vec)  # load pretrained
        #    self.embedding.weight.requires_grad = False #non-trainable
        self.encoder = nn.LSTM(config['emb_dim'], config['hidden_dim'], num_layers=config['nlayers'],
                               bidirectional=config['bidir'], dropout=config['dropout'])
        self.fc = nn.Linear(config['hidden_dim'], config['hidden_fc'])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config['hidden_fc'], config['out_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        # self.hidden = nn.Parameters(self.batch_size, self.hidden_dim)

    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        # M = torch.tanh(encoder_out)
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        # print (wt.shape, new_hidden.shape)
        # new_hidden = torch.tanh(new_hidden)
        # print ('UP:', new_hidden, new_hidden.shape)

        return new_hidden

    def embed(self, sentence_batch):
        inputx = self.dropout(sentence_batch)
        output, (hn, cn) = self.encoder(inputx)
        fbout = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]  # sum bidir outputs F+B
        fbout = fbout.permute(1, 0, 2)
        fbhn = (hn[-2, :, :] + hn[-1, :, :]).unsqueeze(0)
        # print (fbhn.shape, fbout.shape)
        attn_out = self.attnetwork(fbout, fbhn)
        # attn1_out = self.attnetwork1(output, hn)
        score = self.fc2(self.relu(self.fc(attn_out)))

        return score

    def forward(self, s0_batch, s1_batch, lens0, lens1):
        emb_input0 = self.embedding(s0_batch.reshape(s0_batch.shape[1], s0_batch.shape[0]))
        emb_input1 = self.embedding(s1_batch.reshape(s1_batch.shape[1], s1_batch.shape[0]))
        score0 = self.embed(emb_input0)
        score1 = self.embed(emb_input1)
        e0 = torch.exp(score0)
        e1 = torch.exp(score1)
        output = e0 / (e0 + e1)
        return output


class RNNModel(nn.Module):

    def vanilla_rnn_forward(self, batch: torch.Tensor) -> torch.Tensor:
        # output.shape = (batch_size, seq_len, hidden_size)
        # hn.shape = (num_layers, batch_size, hidden_size)
        output, hn = self.rnn(batch)
        h1 = hn[-1].reshape((hn.shape[1], hn.shape[2]))
        return h1

    def lstm_forward(self, batch: torch.Tensor) -> torch.Tensor:
        # output.shape = (batch_size, seq_len, hidden_size)
        # hn.shape = cn.shape = (num_layers, batch_size, hidden_size)
        output, (hn, cn) = self.rnn(batch)
        # samo zadnji layer (-1), reshape da budu prve dvije dim
        h1 = hn[-1].reshape((hn.shape[1], hn.shape[2]))
        return h1

    rnn_cells = {'vanilla': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
    rnn_forwards = {'vanilla': vanilla_rnn_forward, 'lstm': lstm_forward, 'gru': vanilla_rnn_forward}

    def __init__(self, cell_name, embedding, input_size=300, hidden_size=150, num_layers=2, bidirectional=False,
                 dropout=0.0):
        super(RNNModel, self).__init__()
        rnn = RNNModel.rnn_cells[cell_name]
        self.embedding = embedding
        self.rnn = rnn(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       batch_first=True,
                       bidirectional=bidirectional,
                       dropout=dropout
                       )
        self.rnn_forward = RNNModel.rnn_forwards[cell_name]
        self.linear1 = nn.Linear(hidden_size, 150)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(150, 1)

    def forward(self, batch, lens):
        # batch.shape = (batch_size, seq_len, emb_size)
        batch = self.embedding(batch)
        h1 = self.rnn_forward(self, batch)

        h2 = self.relu(self.linear1(h1))
        # h3.shape = (batch_size, 1)
        h3 = self.linear2(h2)
        return h3.reshape((h3.shape[0],))


def train(model, data, optimizer, criterion, epoch, scheduler=None, clip=None):
    model.train()
    for batch_index, batch in enumerate(data):
        s0, s1, y, lens0, lens1 = batch
        model.zero_grad()
        logits = model.forward(s0, s1, lens0, lens1)
        loss = criterion(logits, y.reshape(-1, 1))
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        print(loss.item(), end='\r')
    # print(evaluate(model, data, criterion, epoch))


def evaluate(model, data, criterion, epoch, log_predictions=False):
    model.eval()
    sum_val_loss = 0.0
    counter = 0
    tp, tn, fp, fn = 0, 0, 0, 0

    predictions = None

    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            s0, s1, y, lens0, lens1 = batch
            logits = model(s0, s1, lens0, lens1)

            loss = criterion(logits, y.reshape(-1, 1))
            sum_val_loss += loss
            counter += 1

            y_pred = (logits >= 0.5).int()
            y = y.reshape(-1, 1)

            tp += torch.sum(torch.logical_and(y_pred == 1, y == 1))
            tn += torch.sum(torch.logical_and(y_pred == 0, y == 0))
            fp += torch.sum(torch.logical_and(y_pred == 1, y == 0))
            fn += torch.sum(torch.logical_and(y_pred == 0, y == 1))

            if log_predictions is True:
                if predictions is None:
                    predictions = logits
                else:
                    predictions = torch.vstack((predictions, logits))

    avg_val_loss = sum_val_loss / counter
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    confusion_matrix = torch.tensor([[tp, fp], [fn, tn]])
    dictionary = {
        "loss": avg_val_loss.item(),
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "confusion_matrix": confusion_matrix,
        "epoch": epoch
    }
    if log_predictions is True:
        dictionary['predictions'] = predictions

    return dictionary


def test(model, batch):  # all samples are in this batch
    model.eval()
    with torch.no_grad():
        s0, s1, y, lens0, lens1 = batch
        y = y.reshape(-1, 1)
        batch_p = model(s0, s1, lens0, lens1)
        loss = F.binary_cross_entropy(batch_p, y)
        predictions = batch_p >= 0.5
        acc = torch.sum(predictions == y).item() / len(predictions * 1.0)
        dictionary = {
            'name': None,
            'drop': model.lstm.dropout,
            'epoch_end': None,
            'val_loss': None,
            'val_acc': None,
            'test_acc': acc.item(),
            'test_loss': loss.item(),
            'predictions': batch_p.float().flatten().tolist(),
        }
        return dictionary
