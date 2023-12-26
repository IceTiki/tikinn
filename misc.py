import pathlib as _pathlib
import time as _time
import random as _random
import shutil as _shutil

import numpy as _np
import torch as _torch


def fm_time() -> str:
    time_struct = _time.localtime()
    date_str = (
        f"{time_struct.tm_year}-{time_struct.tm_mon:02d}-{time_struct.tm_mday:02d}"
    )
    time_str = (
        f"{time_struct.tm_hour:02d}-{time_struct.tm_min:02d}-{time_struct.tm_sec:02d}"
    )
    return f"{date_str}_{time_str}"


def indexing_imgfolder(
    folder: str | _pathlib.Path, sample_range: tuple[float, float] = [0, 1]
) -> dict[str, list[_pathlib.Path]]:
    """
    Parameters
    ---
    folder : str | Path
        dataset path
    sample_range : tuple[float, float]
        sample some of image in the folder. If a folder have 0-99 image, [0.2, 0.4] means img 20~39.

    Returns
    ---
    dict[str, list[Path]]
        key is label, value is list of imgpath
    """
    folder = _pathlib.Path(folder)

    tree = {}

    for catetory in folder.iterdir():
        label = catetory.name
        imgs = list(catetory.iterdir())
        a, b = map(lambda x: int(x * len(imgs)), sample_range)

        tree[label] = imgs[a:b]

    return tree


class WorkFolder:
    def __init__(self, work_stores_path: _pathlib.Path, work_name: str = "") -> None:
        self.__work_stores_path: _pathlib.Path = work_stores_path
        self.__work_name: str = (
            f"#t={fm_time()}#h={work_name}##" if work_name else f"#t={fm_time()}##"
        )
        self.__work_path: _pathlib.Path = self.__work_stores_path / self.__work_name

        self.work_root.mkdir(parents=True, exist_ok=True)
        self.model_save_folder.mkdir(parents=True, exist_ok=True)

    @property
    def work_root(self):
        return self.__work_path

    @property
    def log_path(self):
        return self.work_root / "log.log"

    @property
    def model_save_folder(self):
        return self.__work_path / "models"

    def mkdir(self, folder_path: _pathlib.Path) -> _pathlib.Path:
        new_dir = self.__work_path / folder_path
        new_dir.mkdir(exist_ok=True, parents=True)
        return new_dir

    def mkparent(self, file_path: _pathlib.Path) -> _pathlib.Path:
        new_file = self.__work_path / file_path
        new_file.parent.mkdir(exist_ok=True, parents=True)
        return new_file


class SetSeed:
    def __init__(self, seed=114514, deterministic=True) -> None:
        self.set_seed(seed)
        _torch.backends.cudnn.deterministic = deterministic

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        self.set_seed(_time.time())

    @staticmethod
    def set_seed(seed):
        _random.seed(seed)
        _np.random.seed(seed)  # numpy
        _torch.manual_seed(seed)  # CPU
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed(seed)  # GPU
            _torch.cuda.manual_seed_all(seed)  # multi-GPU
