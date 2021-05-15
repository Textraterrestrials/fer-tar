import pytorch_lightning as pl
from torch import nn
from transformers import BertModel


class BertClassification(pl.LightningModule):
    def __init__(self, drop=0.0):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(drop)
        self.linear = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        return self.linear(self.dropout(pooled_output))

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # return optimizer
        pass

    def training_step(self, train_batch, batch_idx):
        # x, y = train_batch
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        # self.log('train_loss', loss)
        # return loss
        pass

    def validation_step(self, val_batch, batch_idx):
        # x, y = val_batch
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        # self.log('val_loss', loss)
        pass


model = BertClassification()
