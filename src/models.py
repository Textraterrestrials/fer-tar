import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from transformers import BertModel, AlbertModel, ElectraModel, RobertaModel


class TransformerBasedClassifier(pl.LightningModule):
    def __init__(self, transformer, embedding_size=None, drop=0.0, **hyperparams):
        super(TransformerBasedClassifier, self).__init__()
        if embedding_size is None:
            embedding_size = transformer.config.pooler_output.shape[1]
        self.transformer = transformer
        self.dropout = nn.Dropout(drop)
        self.linear = nn.Linear(embedding_size, 1)
        self.hyperparams = hyperparams

    def forward(self, x_batch):
        sent0_embedded = self.embed(x_batch[0])
        sent0_embedded = sent0_embedded.mean(dim=2)

        #print(sent0_embedded)
        #print(sent0_embedded.shape)
        sent1_embedded = self.embed(x_batch[1])
        sent1_embedded = sent1_embedded.mean(dim=2)

        bla = self.dropout(sent0_embedded)
        # print(bla.shape)
        sent0_logit = self.linear(bla).reshape(-1, 1)
        sent1_logit = self.linear(self.dropout(sent1_embedded)).reshape(-1, 1)
        e0 = torch.exp(sent0_logit)
        e1 = torch.exp(sent1_logit)
        output = e0 / (e0 + e1)
        return output

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hyperparams.get('lr', 2e-5),
                                eps=self.hyperparams.get('eps', 1e-8),
                                weight_decay=self.hyperparams.get('weight_decay', 0.01))
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=self.hyperparams.get('gamma', 1 - 1e-4))
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, train_batch, batch_idx):
        x_batch, y_batch = train_batch
        return F.binary_cross_entropy(self(x_batch), y_batch)

    def validation_step(self, val_batch, batch_idx):
        x_batch, y_batch = val_batch
        batch_p = self(x_batch)
        loss = F.binary_cross_entropy(batch_p, y_batch)
        predictions = batch_p >= 0.5
        acc = torch.tensor((predictions == y_batch).float()).mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def embed(self, sent_batch):
        pass


class BertClassifier(TransformerBasedClassifier):

    def __init__(self, name='bert-base-uncased', **hyperparams):
        super(BertClassifier, self).__init__(BertModel.from_pretrained(name), **hyperparams)

    def embed(self, sent_batch):
        return self.transformer(
            input_ids=sent_batch['input_ids'],
            attention_mask=sent_batch['attention_mask']
        )[1]


class ElectraClassifier(TransformerBasedClassifier):
    def __init__(self, name='google/electra-small-discriminator', **hyperparams):
        transformer = ElectraModel.from_pretrained(name) # transformer.config.hidden_size
        super(ElectraClassifier, self).__init__(transformer,  embedding_size=256, **hyperparams)

    def embed(self, sent_batch):
        outputs = self.transformer(
            input_ids=sent_batch['input_ids'],
            attention_mask=sent_batch['attention_mask']
        )

        return outputs.last_hidden_state[2]


class AlbertClassifier(TransformerBasedClassifier):

    def __init__(self, name='albert-base-v1', **hyperparams):
        super(AlbertClassifier, self).__init__(AlbertModel.from_pretrained(name), 768, **hyperparams)

    def embed(self, sent_batch):
        outputs = self.transformer(
            input_ids=sent_batch['input_ids'],
            attention_mask=sent_batch['attention_mask']
        )
        # print('vars(outputs)', vars(outputs))
        # print('pooler_output.shape', outputs.pooler_output.shape)
        return outputs.pooler_output


class RobertaClassifier(TransformerBasedClassifier):
    def __init__(self, name='albert-base-v1', **hyperparams):
        super(RobertaClassifier, self).__init__(RobertaModel.from_pretrained(name), 768, **hyperparams)

    def embed(self, sent_batch):
        outputs = self.transformer(
            input_ids=sent_batch['input_ids'],
            attention_mask=sent_batch['attention_mask']
        )
        # print('vars(outputs)', vars(outputs))
        # print('pooler_output.shape', outputs.pooler_output.shape)
        return outputs.pooler_output