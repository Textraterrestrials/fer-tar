import pytorch_lightning as pl
from torch.utils.data import DataLoader

from my_scripts import data
from src import encoders
from src.models import BertClassifier, AlbertClassifier

# example how to load datasets and iterate in batches
# fake_encoder = encoders.FakeEncoder()
# train_ds, dev_ds, test_ds = data.ComVEDataset.from_data_folder(
#     # change this accordingly
#     '~/Documents/tar-data',
#     x_transforms=[encoders.BertEncoder],
#     lazy=True
# )

train_ds = data.ComVEDataset.from_csv(
    'subtaskA_data_all.csv',
    'subtaskA_answers_all.csv',
    x_transforms=[encoders.AlbertEncoder()],
    lazy=True
)
train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)

model = AlbertClassifier()
model.to('cpu')

trainer = pl.Trainer(max_epochs=5, gpus=1, checkpoint_callback=False)
trainer.fit(model, train_dataloader)
