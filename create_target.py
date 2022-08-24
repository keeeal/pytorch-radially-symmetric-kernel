
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch
from torch import cuda
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm

from utils.model import Gauss2d, Sum


def main(input_dir: Path, output_dir: Optional[Path]):

    # Detect whether a CUDA GPU is available.
    device = "cuda" if cuda.is_available() else "cpu"

    # Load data.
    dataset = ImageFolder(input_dir, transform=ToTensor())
    output_dir = output_dir or Path(str(input_dir) + "_target")
    to_image = ToPILImage()

    # Build the model and put it on the chosen device.
    model = Sum(
        Gauss2d(channels=3, kernel_size=[27, 27], alpha=0.25, sigma=10),
        Gauss2d(channels=3, kernel_size=[27, 27], alpha=0.75, sigma=3.14),
    ).to(device)

    # Use the model to create a target dataset.
    with torch.no_grad():
        for (x, _), (file, _) in zip(tqdm(dataset), dataset.samples):
            output_file = output_dir / Path(file).relative_to(dataset.root)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output = model(x.to(device).unsqueeze(0))
            image = to_image(output.squeeze())
            image.save(output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=Path, required=True)
    parser.add_argument("--output-dir", "-o", type=Path)
    main(**vars(parser.parse_args()))
