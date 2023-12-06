"""
for dataset augment
cause of I/O performance, some of augment will be done in the dataset folder
"""
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as trans
import cv2
import tqdm

try:
    from tikilib import image as ti  # my library, for adapting utf-8 path

    load_img = ti.CvIo.load
except ImportError:
    load_img = cv2.imread


def trivial_augment_wide(img: np.ndarray) -> np.ndarray:
    img = np.transpose(img, (2, 0, 1))
    img: torch.Tensor = torch.tensor(img, dtype=torch.uint8)
    img = img.squeeze(0)

    img = trans.TrivialAugmentWide()(img)

    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


def raw_img(img: np.ndarray) -> np.ndarray:
    return img


def transvert_dataset(
    path_in: str | Path,
    path_out: str | Path,
    transvert_method=trivial_augment_wide,
    expand: int = 1,
    sample_range: tuple[float, float] = [0, 1],
    width_limit: int = 300,
):
    """
    Parameters
    ---
    sample_range : tuple[float, float]
        sample some of image in the folder. If a folder have 0-99 image, [0.2, 0.4] means img 20~39.
    """
    path_in, path_out = map(Path, (path_in, path_out))
    for label, paths in tqdm.tqdm(
        misc.indexing_imgfolder(path_in, sample_range).items(),
        dynamic_ncols=True,
        leave=True,
        desc=path_in.name,
    ):
        folder_path = path_out / label
        folder_path.mkdir(parents=True, exist_ok=True)

        new_img_idx = 0
        for img_path in tqdm.tqdm(paths, desc=label, leave=False, position=1):
            img = load_img(img_path)

            for _ in range(expand):
                img_high, img_width, _ = img.shape
                new_width = int(img_width * width_limit / min(img_high, img_width))
                new_high = int(img_high * width_limit / min(img_high, img_width))

                img = cv2.resize(img, (new_width, new_high))

                aug_img = transvert_method(img)

                ti.CvIo.write(aug_img, folder_path / f"{new_img_idx}.jpg")
                new_img_idx += 1


def augment_dataset(
    from_dataset: str | Path,
    new_dataset: str | Path,
    transvert_method=trivial_augment_wide,
    expand: int = 20,
    train_percent: float = 0.8,
    width_limit: int = 300,
):
    from_dataset, new_dataset = map(Path, (from_dataset, new_dataset))
    transvert_dataset(
        from_dataset,
        new_dataset / "train",
        transvert_method,
        expand,
        [0, train_percent],
        width_limit,
    )  # train
    transvert_dataset(
        from_dataset,
        new_dataset / "val",
        lambda x: x,
        1,
        [train_percent, 1],
        width_limit,
    )  # test
