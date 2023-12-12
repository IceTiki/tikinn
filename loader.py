import typing as _typing

from pathlib import Path
import random
from itertools import chain

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, BatchSampler


class FixedNumberDataset(Dataset):
    _t_enbadded_trans: _typing.TypeAlias = _typing.Literal[
        "no_crop", "no_crop_aug", "keep_ratio", "filp_aug", "aff_aug", "auto_aug"
    ]

    @staticmethod
    def get_trans(key: _t_enbadded_trans):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        match key:
            case "no_crop":
                return transforms.Compose(
                    [
                        transforms.Resize([224, 224]),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            case "no_crop_aug":
                return transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.Resize([224, 224]),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            case "keep_ratio":
                return transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            case "filp_aug":
                return transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            case "aff_aug":
                return transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.RandomAffine(
                            degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)
                        ),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            case "auto_aug":
                return transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.AutoAugment(),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )

    @classmethod
    def load_imgfolder(
        cls,
        root: str,
        transform: _typing.Optional[_typing.Callable | _t_enbadded_trans] = None,
        target_transform: _typing.Optional[_typing.Callable] = None,
        loader: _typing.Callable[[str], _typing.Any] = default_loader,
        is_valid_file: _typing.Optional[_typing.Callable[[str], bool]] = None,
        fixed_length: int = 7000,
        sample: _typing.Literal["cycle", "random"] = "cycle",
    ):
        if isinstance(transform, str):
            transform = cls.get_trans(transform)
        dataset = datasets.ImageFolder(
            root, transform, target_transform, loader, is_valid_file
        )
        return cls(dataset, fixed_length, sample)

    def __init__(
        self,
        dataset: Dataset,
        fixed_length: int | None = 7000,
        sample: _typing.Literal["cycle", "random"] = "cycle",
    ) -> None:
        super().__init__()
        self.dataset: Dataset = dataset
        self.fixed_length: int = self.raw_len if fixed_length is None else fixed_length
        self.sample: str = sample

    @property
    def raw_len(self):
        return len(self.dataset)

    def __len__(self):
        return self.fixed_length

    def __getitem__(self, index: int):
        raw_len = self.raw_len
        if index >= self.fixed_length:
            raise ValueError(f"index out of range")
        match self.sample:
            case "cycle":
                return self.dataset[index * (raw_len - 1) // (self.fixed_length - 1)]
            case "random":
                return self.dataset[random.randint(0, raw_len - 1)]
            case x:
                raise ValueError(x)



class MultiDataset(Dataset):
    """
    If you input 3 dataset, len will be the "length of minimum dataset" * 3.
    And the __getitem__ order is dataset_1[0], dataset_2[0], dataset_3[0], dataset_1[1], dataset_2[1], dataset_3[1], ...
    """

    def __init__(self, *datasets_: Dataset) -> None:
        self.__datasets: tuple[Dataset] = datasets_

    @property
    def length_of_minimum_dataset(self) -> int:
        return min((len(i) for i in self.__datasets))

    @property
    def num_dataset(self) -> int:
        return len(self.__datasets)

    @property
    def num_data(self) -> int:
        return self.length_of_minimum_dataset * self.num_dataset

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, index: int):
        num_dataset = self.num_dataset

        ds_idx = index % num_dataset
        item_idx = (index - ds_idx) // num_dataset

        return self.__datasets[ds_idx][item_idx]


class MultiBatchSampler(BatchSampler):
    def __init__(
        self, data_source: MultiDataset, batch_size: int, drop_last: bool, shuffle=True
    ) -> None:
        """
        total batchsize = data_source.num_dataset * batch_size
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.idxs_list: list[list[int]] = [
            list(
                range(
                    i,
                    data_source.num_data,
                    data_source.num_dataset,
                )
            )
            for i in range(data_source.num_dataset)
        ]
        if shuffle:
            for i in self.idxs_list:
                random.shuffle(i)

    def __iter__(self):
        lomd = self.data_source.length_of_minimum_dataset
        batch_size = self.batch_size

        for i in range(0, lomd - (lomd % batch_size), batch_size):
            left = i
            right = i + batch_size
            yield list(chain(*(idxs[left:right] for idxs in self.idxs_list)))

        last = lomd % batch_size
        if last != 0 and not self.drop_last:
            left = lomd - last
            right = lomd
            yield list(chain(*(idxs[left:right] for idxs in self.idxs_list)))

    def __len__(self):
        if self.drop_last:
            return self.data_source.length_of_minimum_dataset // self.batch_size
        return (
            self.data_source.length_of_minimum_dataset + self.batch_size - 1
        ) // self.batch_size
