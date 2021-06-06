import torch
from torch import nn
from torch.nn import functional as F


class LSTMModel(nn.Module):
    """bidirectional and batch_first."""

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
