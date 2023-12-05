from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    TypeAlias,
)
from pathlib import Path
import random
from itertools import chain

import torch
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

from .constants import Directions


def get_model(
    name: Literal["resnet50", "resnet50_pre"]
    | tuple[Literal["eval", "dict"], str, Path] = "resnet50",
    grad: Literal[None, "only_fc"] = None,
    num_classes=65,
):
    match name:
        case "resnet50_pre":
            params = torch.load(Directions.pretrained_models / "resnet50-19c8e357.pth")
            model = models.resnet50()
            model.load_state_dict(params)
            if not model.fc.out_features == num_classes:
                cnn_features = model.fc.in_features
                model.fc = torch.nn.Sequential(
                    nn.Linear(cnn_features, num_classes, bias=True).cuda()
                )
        case str():
            model = getattr(models, name)(num_classes=num_classes)
        case "eval", str(), Path():
            _, arch, path_ = name
            model: nn.Module = torch.load(path_)
            model.eval()
        case "dict", str(), Path():
            _, arch, path_ = name
            params = torch.load(path_)
            model = getattr(models, arch)()
            model.load_state_dict(params)
        case _:
            print(name)
            raise ValueError()

    match grad:
        case None:
            pass
        case "only_fc":
            for param in model.parameters():
                param.requires_grad = False
            cnn_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(cnn_features, num_classes, bias=True))

    model = model.cuda()
    return model


def get_dataloader(
    name: Literal[
        "sa_art",
        "sa_clipart",
        "sa_product",
        "sa_realworld",
        "t_art",
        "t_product",
        "t_realworld",
        "t_clipart",
        "art2product",
        "art2realworld",
        "clipart2art",
    ],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    match name:
        case "t_art":
            dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "art_test",
                "no_crop",
                fixed_length=None,
                sample="cycle",
            )
            dataloader = _as_one_dataloader(dataset, batch_size, num_workers, True)
        case "t_product":
            dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "product_test",
                "no_crop",
                fixed_length=None,
                sample="cycle",
            )
            dataloader = _as_one_dataloader(dataset, batch_size, num_workers, True)
        case "t_realworld":
            dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "realworld_test",
                "no_crop",
                fixed_length=None,
                sample="cycle",
            )
            dataloader = _as_one_dataloader(dataset, batch_size, num_workers, True)
        case "t_clipart":
            dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "clipart_test",
                "no_crop",
                fixed_length=None,
                sample="cycle",
            )
            dataloader = _as_one_dataloader(dataset, batch_size, num_workers, True)

        case "art2product":
            source_dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "art_train_e5_a",
                "filp_aug",
                fixed_length=7000,
                sample="random",
            )
            target_dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "product_test",
                "no_crop",
                fixed_length=7000,
                sample="random",
            )
            dataloader = _as_multi_dataloader(
                source_dataset,
                target_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        case "art2realworld":
            source_dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "art_train_e5_a",
                "filp_aug",
                fixed_length=7000,
                sample="random",
            )
            target_dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "realworld_test",
                "no_crop",
                fixed_length=7000,
                sample="random",
            )
            dataloader = _as_multi_dataloader(
                source_dataset,
                target_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        case "clipart2art":
            source_dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "clipart_train_e5_a",
                "no_crop_aug",
                fixed_length=7000,
                sample="random",
            )
            target_dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "art_test",
                "no_crop",
                fixed_length=7000,
                sample="random",
            )
            dataloader = _as_multi_dataloader(
                source_dataset,
                target_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
            )

        case "sa_art":
            dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "art_train_e5_a",
                "filp_aug",
                fixed_length=7000,
                sample="random",
            )
            dataloader = _as_one_dataloader(dataset, batch_size, num_workers, True)
        case "sa_clipart":
            dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "clipart_train_e5_a",
                "no_crop_aug",
                fixed_length=7000,
                sample="random",
            )
            dataloader = _as_one_dataloader(dataset, batch_size, num_workers, True)
        case "sa_product":
            dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "product_train_e5_a",
                "filp_aug",
                fixed_length=7000,
                sample="random",
            )
            dataloader = _as_one_dataloader(dataset, batch_size, num_workers, True)
        case "sa_realworld":
            dataset = FixedNumberDataset.load_imgfolder(
                Directions.dataset / "realworld_train_e5_a",
                "filp_aug",
                fixed_length=7000,
                sample="random",
            )
            dataloader = _as_one_dataloader(dataset, batch_size, num_workers, True)

        case err:
            raise ValueError(str(err))

    return dataloader


def get_optimizer(
    model: models.ResNet,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    flag: str = "half_conv_learning_rate",
):
    if flag == "half_conv_learning_rate":
        return torch.optim.SGD(
            (
                {
                    "params": (
                        filter(
                            lambda x: (
                                all(i is not x for i in model.fc.parameters())
                                and x.requires_grad
                            ),
                            model.parameters(),
                        )
                    )
                },
                {"params": model.fc.parameters(), "lr": learning_rate},
            ),
            learning_rate / 10,
            momentum=momentum,
            weight_decay=weight_decay,
        )


def _as_one_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle=True
) -> DataLoader:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
    )
    return dataloader


def _as_multi_dataloader(*datasets_: Dataset, batch_size: int, num_workers: int):
    dataset = MultiDataset(*datasets_)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=MultiBatchSampler(dataset, batch_size, drop_last=False),
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


class FixedNumberDataset(Dataset):
    _t_enbadded_trans: TypeAlias = Literal[
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
        transform: Optional[Callable | _t_enbadded_trans] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        fixed_length: int = 7000,
        sample: Literal["cycle", "random"] = "cycle",
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
        sample: Literal["cycle", "random"] = "cycle",
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
        match self.sample:
            case "cycle":
                return self.dataset[index * (self.fixed_length - 1) // (raw_len - 1)]
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
