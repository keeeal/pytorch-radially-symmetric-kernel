
from pathlib import Path
from typing import Iterable

from torch import Tensor, get_num_threads
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import DatasetFolder, ImageFolder


def get_relative_files(dataset: DatasetFolder) -> list[str]:
    return [Path(file).relative_to(dataset.root) for file, _ in dataset.samples]


class PairedImageFolders(Dataset):
    def __init__(self, input_root: Path, target_root: Path, *args, **kwargs) -> None:
        self.input_dataset = ImageFolder(input_root, *args, **kwargs)
        self.target_dataset = ImageFolder(target_root, *args, **kwargs)

        input_files = get_relative_files(self.input_dataset)
        target_files = get_relative_files(self.target_dataset)
        assert input_files == target_files

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.input_dataset[index][0], self.target_dataset[index][0]

    def __len__(self) -> int:
        return len(self.input_dataset)
