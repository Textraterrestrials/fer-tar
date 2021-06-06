import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from transformers import AutoModel, get_linear_schedule_with_warmup

MODELS = {
    'bert': 'distilbert-base-uncased',
    'albert': 'albert-base-v1',
    'roberta': 'distilroberta-base',
    'electra': 'google/electra-base-discriminator'
}


class TransformerBasedClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = kwargs.get('name', 'bert')
        self.transformer = AutoModel.from_pretrained(MODELS[self.name])
        self.dropout = nn.Dropout(kwargs.get('drop', 0.1))
        self.linear = nn.Linear(self.transformer.config.hidden_size, 1)
        self.kwargs = kwargs
        self.val_losses = []
        self.val_accuracies = []

    def forward(self, x_batch):
        sent0_embedded = self.embed(x_batch[0])
        sent1_embedded = self.embed(x_batch[1])

        sent0_logits = self.linear(self.dropout(sent0_embedded)).reshape(-1, 1)
        sent1_logits = self.linear(self.dropout(sent1_embedded)).reshape(-1, 1)

        e0 = torch.exp(sent0_logits)
        e1 = torch.exp(sent1_logits)
        return e0 / (e0 + e1)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.kwargs.get('lr', 1e-4),
            eps=self.kwargs.get('eps', 1e-8),
            weight_decay=self.kwargs.get('weight_decay', 0.01)
        )
        scheduler = {'scheduler': get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=self.kwargs.get('training_steps'),
            num_warmup_steps=self.kwargs.get('training_steps') // 8),
            'interval': 'step'
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, train_batch, batch_idx):
        x_batch, y_batch = train_batch
        batch_p = self(x_batch)
        loss = F.binary_cross_entropy(batch_p, y_batch)
        return loss

    def get_best_loss_and_acc(self):
        loss_idx = np.argmin(self.val_losses)
        return self.val_losses[loss_idx], self.val_accuracies[loss_idx]

    def validation_step(self, val_batch, batch_idx):  # all samples are in this batch
        x_batch, y_batch = val_batch
        batch_p = self(x_batch)
        loss = F.binary_cross_entropy(batch_p, y_batch)
        predictions = batch_p >= 0.5
        acc = torch.sum(predictions == y_batch).item() / len(predictions * 1.0)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.val_losses.append(loss.item())
        self.val_accuracies.append(acc)
        # self.val_losses.append(loss.item())
        # print(self.val_losses)

    def test_step(self, test_batch, batch_idx):  # all samples are in this batch
        x_batch, y_batch = test_batch
        batch_p = self(x_batch)
        loss = F.binary_cross_entropy(batch_p, y_batch)
        predictions = batch_p >= 0.5
        acc = torch.sum(predictions == y_batch) / len(predictions * 1.0)
        dictionary = {
            'name': self.transformer.config.name_or_path,
            'drop': self.dropout.p,
            'epoch_end': self.current_epoch + 1,
            'val_loss': self.val_loss,
            'val_acc': self.val_acc,
            'test_acc': acc.item(),
            'test_loss': loss.item(),
            'predictions': batch_p.flatten().tolist(),

        }
        append_name = self.kwargs.get('append_name', 'plain')
        # result_file = f'{self.name}-drop={self.dropout.p}-{append_name}.json'
        result_file = f'{self.name}-{append_name}.json'
        with open(result_file, 'w') as out:
            import json
            out.write(json.dumps(dictionary, indent=4))
        self.log_dict({'test_loss': loss, 'test_acc': acc})

    def embed(self, sent_batch):
        return self.transformer(
            input_ids=sent_batch['input_ids'],
            attention_mask=sent_batch['attention_mask']
        ).last_hidden_state.mean(dim=1)  # (batch_size, sequence_length, hidden_size), we want CLS token only


class PseudoEnsemble:
    def __init__(self, prediction_dir='.'):
        self.prediction_dir = prediction_dir

    def majority(self):
        combination = []
        for file in os.listdir(self.prediction_dir):
            if file.endswith('.json'):
                with open(self.prediction_dir + '/' + file) as f:
                    predictions = np.array(json.load(f)['predictions']) >= 0.5
                    predictions = 2 * predictions - 1
                    combination.append(predictions)

        return np.sum(combination, axis=0) > 0

    def maximum(self):
        combination = []
        for file in os.listdir():
            if file.endswith('.json'):
                with open(file) as f:
                    predictions = np.array(json.load(f)['predictions'])
