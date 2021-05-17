import pytorch_lightning as pl
from torch.utils.data import DataLoader

from my_scripts import data
from src.modelss import TransformerBasedClassifier
from src.tokenizers import TransformerTokenizer

NAME = 'bert'

train_ds = data.ComVEDataset.from_csv(
    'Training/subtaskA_data_all.csv',
    'Training/subtaskA_answers_all.csv',
    x_transforms=[TransformerTokenizer(NAME)],
    lazy=True
)
train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)

model = TransformerBasedClassifier(NAME)

trainer = pl.Trainer(fast_dev_run=True, max_epochs=5, checkpoint_callback=False)
trainer.fit(model, train_dataloader)
