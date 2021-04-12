from torch import utils
from scripts import data
from src import encoders
# example how to load datasets and iterate in batches
fake_encoder = encoders.FakeEncoder()
train_ds, dev_ds, test_ds = data.ComVEDataset.from_data_folder(
    # change this accordingly
    '~/Documents/tar-data',
    x_transforms=[fake_encoder],
    lazy=True
)

train_dataloader = utils.data.DataLoader(train_ds, batch_size=6, shuffle=True)

for i, batch in enumerate(train_dataloader):
    x_batch, y_batch = batch
    print(f"batch {i}")
    print(x_batch)
    print(y_batch)
    break
