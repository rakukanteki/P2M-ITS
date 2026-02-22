import torch
from torch.utils.data import DataLoader
from ..data.dataset import FolderDataset


class FLClient:
    def __init__(self, client_id: int, samples, config):
        self.client_id = client_id
        self.dataset = FolderDataset(samples)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.training.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
