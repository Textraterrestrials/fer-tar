import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from transformers import AutoModel

MODELS = {
    'bert': 'bert-base-uncased',
    'albert': 'albert-base-v1',
    'roberta': 'roberta-base',
    'electra': 'google/electra-small-discriminator'
}


class TransformerBasedClassifier(pl.LightningModule):
    def __init__(self, name='bert', **kwargs):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(MODELS[name])
        self.dropout = nn.Dropout(kwargs.get('drop', 0.1))
        self.linear = nn.Linear(self.transformer.config.hidden_size, 1)
        self.kwargs = kwargs

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
            lr=self.kwargs.get('lr', 2e-5),
            eps=self.kwargs.get('eps', 1e-8),
            weight_decay=self.kwargs.get('weight_decay', 0.01)
        )
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.kwargs.get('gamma', 1 - 1e-4)
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, train_batch, batch_idx):
        x_batch, y_batch = train_batch
        batch_p = self(x_batch)
        loss = F.binary_cross_entropy(batch_p, y_batch)
        return loss

    def validation_step(self, val_batch, batch_idx):  # all samples are in this batch
        x_batch, y_batch = val_batch
        batch_p = self(x_batch)
        loss = F.binary_cross_entropy(batch_p, y_batch)
        predictions = batch_p >= 0.5
        acc = torch.sum(predictions == y_batch).item() / len(predictions * 1.0)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, test_batch, batch_idx):  # all samples are in this batch
        x_batch, y_batch = test_batch
        batch_p = self(x_batch)
        loss = F.binary_cross_entropy(batch_p, y_batch)
        predictions = batch_p >= 0.5
        acc = torch.sum(predictions == y_batch).item() / len(predictions * 1.0)
        dictionary = {
            'drop': self.dropout.p,
            'epoch_end': self.current_epoch + 1,
            'predictions': predictions.long().flatten().tolist(),
            'acc': acc,
            'loss': loss.item()
        }
        with open(self.transformer.config.name_or_path + '.conf', 'w') as out:
            import json
            out.write(json.dumps(dictionary, indent=4))

    def embed(self, sent_batch):
        return self.transformer(
            input_ids=sent_batch['input_ids'],
            attention_mask=sent_batch['attention_mask']
        ).last_hidden_state[:, 0, :]  # (batch_size, sequence_length, hidden_size), we want CLS token only
