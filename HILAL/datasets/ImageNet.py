import os
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from .Dataset import Dataset
from .transforms import Cutout

class ImageNet(Dataset):
    def __init__(self, train_indices: torch.Tensor = None, valid_indices: torch.Tensor = None):
        self.train_indices = train_indices
        self.valid_indices = valid_indices

    def load_train_data(
        self,
        batch_size: int,
        num_workers: int,
        validation: bool,
    ) -> DataLoader:
        if not validation:
            assert self.valid_indices is None

        if self.train_indices is not None or self.valid_indices is not None:
            assert self.train_indices is not None and self.valid_indices is not None

        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                Cutout(1, length=56),
            ]
        )

        dataset = torchvision.datasets.ImageNet(
            #change here
            root=os.path.join(os.getcwd(), "data", "imagenet"),
            split="train",
            transform=transform,
        )

        if validation:
            train_size = int(0.9 * len(dataset))
            valid_size = int(0.1 * len(dataset))
        else:
            train_size = len(dataset)
            valid_size = 0

        if valid_size > 0:
            if self.train_indices is None:
                train_dataset, valid_dataset = torch.utils.data.random_split(
                    dataset, [train_size, valid_size]
                )
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    train_dataset.indices
                )
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    valid_dataset.indices
                )
            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_indices)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.valid_indices)

            valid_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=valid_sampler,
            )
        else:
            train_sampler = None
            valid_loader = None

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
        )

        if validation:
            return valid_loader
        else:
            return train_loader

    def load_test_data(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoader:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        val_dataset = torchvision.datasets.ImageNet(
            #change here
            root=os.path.join(os.getcwd(), "data", "imagenet"),
            split="val",
            transform=transform,
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        return val_loader