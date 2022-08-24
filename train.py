
from argparse import ArgumentParser
from itertools import count, islice
from pathlib import Path
from typing import Iterable

from torch import cuda, device, nn, Tensor, get_num_threads
from torch.nn.modules.loss import _Loss
from torch.optim import Adam, Optimizer, SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from tqdm import tqdm

from utils.data import PairedImageFolders
from utils.model import Gauss2d, Sum


def print_parameters(model: nn.Module):
    state = model.state_dict()
    for key in state:
        if key.endswith(("alpha", "sigma")):
            print(key, "=", state[key].item())


def train(
    model: nn.Module,
    data: Iterable[tuple[Tensor, Tensor]],
    loss_fn: _Loss,
    optimizer: Optimizer,
    device: device | str,
) -> float:
    model.train()
    losses = []

    for item, target in tqdm(data):
        optimizer.zero_grad()
        item, target = item.to(device), target.to(device)
        loss = loss_fn(model(item), target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return sum(losses) / len(losses)


def main(
    input_dir: Path,
    target_dir: Path,
    learn_rate: float,
):
    # Detect whether a CUDA GPU is available.
    device = "cuda" if cuda.is_available() else "cpu"

    # Load data.
    print("\nLoading data...")
    train_data = DataLoader(
        PairedImageFolders(input_dir, target_dir, transform=ToTensor()),
        batch_size=1,
        shuffle=True,
        num_workers=get_num_threads(),
    )
    print(f"Training data size: {len(train_data.dataset)}")

    # Build the model and put it on the chosen device.
    print("\nBuilding model...")
    model = Sum(
        Gauss2d(channels=3, kernel_size=[27, 27]),
        Gauss2d(channels=3, kernel_size=[27, 27]),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(n_params, "parameters.")

    # Create the loss function and optimizer.
    loss_fn = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=learn_rate, momentum=0.9)
    # optimizer = Adam(model.parameters(), lr=learn_rate)

    # Print current parameters
    print_parameters(model)
    print()

    for epoch in count():
        some_train_data = islice(train_data, 100)
        loss = train(model, some_train_data, loss_fn, optimizer, device)
        print(f"{epoch = }")
        print(f"{loss = :.2e}")

        # Print current parameters
        print_parameters(model)
        print()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=Path, required=True)
    parser.add_argument("-t", "--target-dir", type=Path, required=True)
    parser.add_argument("-lr", "--learn-rate", type=float, default=0.1)
    main(**vars(parser.parse_args()))
