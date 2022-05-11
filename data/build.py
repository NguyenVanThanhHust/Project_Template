from torch.utils import data
from .datasets import CrackDataset
from .transforms import build_transforms

def build_datasets(data_folder, transform, is_train=True):
    datasets = CrackDataset(data_folder=data_folder, split=split, transform=transform)
    return datasets

def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False
    transform = build_transforms(cfg, is_train=is_train)
    datasets = build_datasets(cfg.INPUT.FOLDER, transform, is_train)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader