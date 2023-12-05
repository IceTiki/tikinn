import pathlib as _pathlib


class Directions:
    repository_root = _pathlib.Path(__file__).parent.parent
    dataset = repository_root / "dataset"
    office = dataset / r"OfficeHomeDataset_10072016"
    office_art = office / "Art"
    office_clipart = office / "Clipart"
    office_product = office / "Product"
    office_realworld = office / "Real World"
    pretrained_models = repository_root / "pretrained_models"
